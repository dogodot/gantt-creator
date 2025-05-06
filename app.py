import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
from datetime import datetime
import re
import openai
from dotenv import load_dotenv
import os

# Set page to full width
st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()

# Attercop logo at the top left, not clickable
col_logo, _ = st.columns([0.1, 0.9])
with col_logo:
    st.image("static/attercop_logo.png", use_container_width=True)

# Main title lower on the page
st.title("Attercop Gantt Generator")

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
        'Task ID', 'Task Name', 'Phase', 'Duration (Weeks)', 'Start Date', 'End Date', 'Predecessors', 'Deliverables Mapping'
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
            resource = 'Group Header' if is_group else row['Phase']
            gantt_data.append({
                'Task': label,
                'Start': start_date,
                'Finish': end_date,
                'Resource': resource,
                'Bold': is_group
            })
        except ValueError:
            continue
    # Define colors for different phases and group headers
    colors = {
        'Group Header': 'rgb(0,0,0)',  # Black for group headers
        'Discovery': 'rgb(46, 137, 205)',
        'Strategy Definition': 'rgb(114, 44, 121)',
        'Use Case Definition': 'rgb(198, 47, 105)',
        'Planning & Requirements': 'rgb(58, 149, 136)',
        'Finalisation': 'rgb(107, 127, 135)'
    }
    fig = ff.create_gantt(pd.DataFrame(gantt_data), 
                          colors=colors,
                          index_col='Resource',
                          show_colorbar=True,
                          group_tasks=True,
                          showgrid_x=True,
                          showgrid_y=True,
                          height=800,
                          title='Project Timeline Gantt Chart')
    fig.update_layout(
        title_x=0.5,
        title_font_size=24,
        font=dict(size=12),
        xaxis_title="Date",
        yaxis_title="Tasks",
        showlegend=True,
        width=None
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