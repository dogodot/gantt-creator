import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
from datetime import datetime
import re
import openai
from dotenv import load_dotenv
import os
import random

# Set page to full width
st.set_page_config(
    page_title="Gantt Chart Generator",
    page_icon="📅",
    layout="wide"
)

# Load environment variables from .env file
load_dotenv()

# Attercop logo at the top left, not clickable
col_logo, _ = st.columns([0.1, 0.9])
with col_logo:
    st.image("static/attercop_logo.png", use_container_width=True)

# Main title lower on the page
st.title("Gantt Chart Generator")

def parse_markdown_table(markdown_text):
    # Split the text into lines and remove empty lines
    lines = [line.strip() for line in markdown_text.strip().split('\n') if line.strip()]
    errors = []
    if not lines:
        errors.append("No input provided. Please paste a markdown table.")
        st.error("\n".join(errors))
        return None
    
    # Find the table header
    header_line = None
    for i, line in enumerate(lines):
        if '|' in line:
            header_line = i
            break
    if header_line is None:
        errors.append("No valid markdown table found. Ensure your table uses '|' to separate columns.")
        st.error("\n".join(errors))
        return None
    
    # Extract headers
    headers = [h.strip().strip('*') for h in lines[header_line].split('|') if h.strip()]
    required_headers = [
        'Task ID', 'Task Name', 'Phase', 'Duration (Days)', 'Start Date', 'End Date', 'Predecessors', 'Deliverables Mapping'
    ]
    missing_headers = [h for h in required_headers if h not in headers]
    if missing_headers:
        errors.append(f"Missing required columns: {', '.join(missing_headers)}. Please ensure your table includes all required columns.")
    
    # Skip the separator line (if it exists)
    data_start = header_line + 2 if header_line + 1 < len(lines) and '|' in lines[header_line + 1] else header_line + 1
    
    # Parse the data
    data = []
    for row_num, line in enumerate(lines[data_start:], start=data_start+1):
        if not line.strip() or '|' not in line:
            continue
        # Split and preserve empty cells, strip * and whitespace
        cells = [cell.strip() for cell in line.split('|')]
        # Remove leading/trailing empty cells if the line starts/ends with '|'
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        # Detect if this row is a group header (bold in markdown)
        is_group_header = False
        # Check if any cell is bold (starts and ends with **)
        for c in cells:
            if c.startswith('**') and c.endswith('**'):
                is_group_header = True
                break
        # Remove bold markers for all cells
        cells = [cell.strip('*').strip() for cell in cells]
        if len(cells) < len(headers):
            errors.append(f"Row {row_num}: Missing cells. Expected {len(headers)} columns, found {len(cells)}.")
            continue
        if len(cells) > len(headers):
            errors.append(f"Row {row_num}: Extra cells. Expected {len(headers)} columns, found {len(cells)}.")
            continue
        # Validate date columns (skip for group headers)
        if not is_group_header:
            for date_col in ['Start Date', 'End Date']:
                idx = headers.index(date_col)
                try:
                    datetime.strptime(cells[idx], '%d/%m/%Y')
                except ValueError:
                    errors.append(f"Row {row_num}: Invalid date format in '{date_col}'. Expected DD/MM/YYYY, got '{cells[idx]}'.")
        # Create task entry
        task_entry = {headers[i]: cells[i] for i in range(len(headers))}
        task_entry['IsGroupHeader'] = is_group_header
        data.append(task_entry)
    if not data:
        errors.append("No valid data rows found in the table.")
    if errors:
        st.error("\n".join(errors))
        return None
    return pd.DataFrame(data)

def create_gantt_chart(df, mode="Detail mode (all lines)"):
    if df is None or df.empty:
        return None

    # First, get the phase order from the group headers
    phase_order = []
    for _, row in df[df['IsGroupHeader']].sort_values('Task ID').iterrows():
        phase_num = row['Task ID'].split('.')[0]  # Get the phase number
        phase_name = row['Phase'].strip().title()
        phase_order.append(f"{phase_num}. {phase_name}")
    
    # Create phase mapping based on the order from the data
    phase_mapping = {}
    for numbered_phase in phase_order:
        phase_name = numbered_phase.split('. ', 1)[1]  # Get the phase name without the number
        phase_mapping[phase_name] = numbered_phase

    # Prepare data for Gantt chart, preserve row order
    gantt_data = []
    for _, row in df.iterrows():
        label = f"{row['Task ID']}: {row['Task Name']}"
        is_group = row.get('IsGroupHeader', False)
        if mode == "Phase Headers only" and not is_group:
            continue
        try:
            start_date = datetime.strptime(row['Start Date'], '%d/%m/%Y').strftime('%Y-%m-%d')
            end_date = datetime.strptime(row['End Date'], '%d/%m/%Y').strftime('%Y-%m-%d')
            # Normalize phase name and add number prefix
            phase = row['Phase'].strip().title()
            numbered_phase = phase_mapping.get(phase, phase)
            gantt_data.append({
                'Task': label,
                'Start': start_date,
                'Finish': end_date,
                'Resource': numbered_phase,
                'Bold': is_group,
                'Duration': row['Duration (Days)']
            })
        except ValueError:
            continue

    # Create DataFrame
    gantt_df = pd.DataFrame(gantt_data)

    # Generate colors for each phase
    colors = {}
    for phase in phase_order:
        # Generate a consistent color for each phase
        random.seed(phase)  # Use phase name as seed for consistent colors
        r = random.randint(0, 200)
        g = random.randint(0, 200)
        b = random.randint(0, 200)
        colors[phase] = f'rgb({r},{g},{b})'

    # Create the Gantt chart
    fig = ff.create_gantt(gantt_df, 
                         colors=colors,
                         index_col='Resource',
                         show_colorbar=True,
                         group_tasks=True,
                         showgrid_x=True,
                         showgrid_y=True,
                         height=800)
    
    # Update layout
    fig.update_layout(
        font=dict(size=12),
        xaxis_title="Date",
        yaxis_title="Tasks",
        showlegend=True,
        width=None,
        legend=dict(
            traceorder='normal',
            itemsizing='constant'
        )
    )

    # Force the y-axis order
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=phase_order
    )
    
    return fig

def dataframe_to_markdown(df):
    if df is None or df.empty:
        return ""
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(val) for val in row) + " |\n"
    return markdown

# Streamlit UI


# Add example markdown with 8 columns per row, even if the last cell is empty
example_markdown = """Task ID | Task Name | Phase | Duration (Weeks) | Start Date | End Date | Predecessors | Deliverables Mapping
1 | Phase 1: Initiation & Discovery | Discovery | 2 | 19/05/2025 | 30/05/2025 |  |  
1.1 | Project Kick-off & Governance Setup | Discovery | 0.5 | 19/05/2025 | 21/05/2025 |  |  |
1.2 | Request & Review Plenitude Documents | Discovery | 1.5 | 19/05/2025 | 28/05/2025 |  | Input: 1, 2, 3, 6 
"""

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'markdown_input' not in st.session_state:
    st.session_state.markdown_input = ""

# Create two columns for the layout with custom widths
col1, col2 = st.columns([0.4, 0.6])  # Adjust the ratio to give more space to the table

with col1:
    st.markdown("### Input/Edit Data")
    st.markdown("You can either paste a markdown table or edit the data directly in the table below.")
    st.markdown("Dates should be in DD/MM/YYYY format")
    
    # Add example button
    if st.button("Load Example"):
        st.session_state.markdown_input = example_markdown
        st.session_state.df = parse_markdown_table(example_markdown)
    
    # Add markdown input text area
    markdown_input = st.text_area("Markdown Table", value=st.session_state.markdown_input, height=300)
    
    # Process markdown input
    if markdown_input != st.session_state.markdown_input:
        st.session_state.df = parse_markdown_table(markdown_input)
        st.session_state.markdown_input = markdown_input

with col2:
    st.markdown("### Editable Table View")
    if st.session_state.df is not None:
        # Display editable dataframe with full width
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            height=400
        )
        
        # Update markdown if table was edited
        if not edited_df.equals(st.session_state.df):
            st.session_state.df = edited_df
            st.session_state.markdown_input = dataframe_to_markdown(edited_df)

# Generate Gantt Chart
st.markdown("---")
view_mode = st.radio(
    "Gantt Chart View Mode:",
    ["Detail mode (all lines)", "Phase Headers only"],
    index=0,
    horizontal=True
)

if st.button("Generate Gantt Chart", use_container_width=True):
    if st.session_state.df is not None and not st.session_state.df.empty:
        fig = create_gantt_chart(st.session_state.df, mode=view_mode)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            # Add download button for the HTML file
            html = fig.to_html(include_plotlyjs="cdn")
            st.download_button(
                label="Download Gantt Chart as HTML",
                data=html,
                file_name="gantt_chart.html",
                mime="text/html",
                use_container_width=True
            )
    else:
        st.warning("Please enter or edit data first")

# --- Chatbot Section at the bottom ---
st.markdown("---")
st.header("Ask questions about your project plan")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get the current markdown data
markdown_data = st.session_state.get("markdown_input", "")

# Chat input
user_message = st.chat_input("Ask a question about your project plan...")

if user_message:
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    # Compose the system prompt and messages
    system_prompt = (
        "You are a helpful assistant. The user will provide a project plan in markdown table format. "
        "Answer questions about the project plan using only the information in the table. "
        "If the answer is not in the table, say you don't know."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the project plan in markdown:\n\n{markdown_data}"},
    ]
    # Add chat history
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Call OpenAI GPT-4 using the new API and environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY environment variable not set. Please create a .env file with your key.")
    else:
        client = openai.OpenAI(api_key=openai_api_key)
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.2,
            )
        assistant_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# ---
# .env file example (do not commit this file to git):
# OPENAI_API_KEY=sk-... 

def convert_to_gantt(plan):
    prompt = """You are an expert Project Manager. Create a markdown gantt chart from this project plan. The plan will have the following columns:
| Task ID | Task Name | Phase | Duration (Weeks) | Start Date | End Date | Predecessors | Deliverables Mapping |
| :------ | :---------------------------------------------------- | :-------------------------------- | :--------------- | :----------- | :----------- | :----------- | :------------------- |

Please follow these specific requirements:
1. Use British English spelling and terminology
2. All dates must be in DD/MM/YYYY format
3. Include Group Header lines for each phase in the following format:
   | **1.0** | **Phase 1: Initiation & Discovery** | **Discovery** | **2** | **19/05/2025** | **30/05/2025** |              |                      |
4. For task IDs, use decimal format (e.g., 1.1, 1.2, 2.1) where the first number represents the phase
5. For predecessors, use the Task IDs (e.g., "1.1, 1.2")
6. For deliverables, provide a brief description of what each task produces
7. IMPORTANT: Phase names must be EXACTLY the same for both headers and tasks. Do not use abbreviations or variations.
   - Use "Planning & Requirements" (not "Planning & Req.")
   - Use "Strategy Definition" (not "Strategy" or "Strategy Def.")
   - Use "Use Case Definition" (not "Use Cases" or "UCD")
   - Use "Discovery" (not "Initiation" or "Discovery Phase")
   - Use "Finalisation" (not "Final" or "Final Phase")

Please analyze the project plan and create a structured markdown table with all tasks, their phases, durations, and dependencies. Include realistic start and end dates based on the current date.

Current date: {current_date}

Project Plan:
{plan}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert Project Manager who creates detailed Gantt charts from project plans. You always use British English and follow the specified formatting requirements exactly. You must use consistent phase names throughout the table."},
                {"role": "user", "content": prompt.format(current_date=datetime.now().strftime("%d/%m/%Y"), plan=plan)}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling GPT-4: {str(e)}")
        return None 