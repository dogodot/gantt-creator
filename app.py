import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import re
import openai
from dotenv import load_dotenv
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Attercop Gantt Generator",
    page_icon="📊",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Attercop logo at the top left, not clickable
col_logo, _ = st.columns([0.1, 0.9])
with col_logo:
    st.image("static/attercop_logo.png", use_container_width=True)

# Main title lower on the page
st.title("Attercop Gantt Generator")

# Instructions for users
st.markdown("""
### Instructions
1. Enter your project plan in the text area below. Describe your project tasks, their durations, dependencies, and deliverables.
2. Click 'Generate Gantt Chart' to convert your plan into a structured Gantt chart.
3. The chart will be displayed below with all tasks, their durations, and dependencies.
""")

# Collapsible example input
with st.expander("Click to see an example input"):
    st.markdown("""
```
Project: Website Redesign
Phase 1: Planning (2 weeks)
- Task 1: Requirements gathering (1 week)
- Task 2: Design approval (1 week)

Phase 2: Development (4 weeks)
- Task 3: Frontend development (2 weeks)
- Task 4: Backend development (3 weeks)
- Task 5: Integration (1 week)

Phase 3: Testing (2 weeks)
- Task 6: Unit testing (1 week)
- Task 7: User acceptance testing (1 week)
```
""")

# Text area for project plan input
project_plan = st.text_area("Enter your project plan:", height=300)

# Function to convert project plan to Gantt chart using GPT-4
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

Please analyze the project plan and create a structured markdown table with all tasks, their phases, durations, and dependencies. Include realistic start and end dates based on the current date.

Current date: {current_date}

Project Plan:
{plan}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert Project Manager who creates detailed Gantt charts from project plans. You always use British English and follow the specified formatting requirements exactly."},
                {"role": "user", "content": prompt.format(current_date=datetime.now().strftime("%d/%m/%Y"), plan=plan)}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling GPT-4: {str(e)}")
        return None

# Function to parse markdown table to DataFrame
def parse_markdown_table(markdown_table):
    # If the input is a list, join it into a single string
    if isinstance(markdown_table, list):
        markdown_table = '\n'.join(markdown_table)
    
    # Split the table into lines and remove empty lines
    lines = [line.strip() for line in markdown_table.split('\n') if line.strip()]
    
    # Find the header line (contains column names)
    header_index = -1
    for i, line in enumerate(lines):
        if '|' in line and any(col in line for col in ['Task ID', 'Task Name', 'Phase']):
            header_index = i
            break
    
    if header_index == -1:
        st.error("Could not find table header. Please ensure the table has the correct column names.")
        return None
    
    # Start processing data from the line after the header
    data_lines = lines[header_index + 1:]
    
    # Parse the data
    data = []
    current_phase = None
    for line in data_lines:
        # Skip separator line if it exists
        if all(c in '-:' for c in line.strip('|')):
            continue
        # Split the line by | and remove empty strings
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if len(cells) >= 7:  # Ensure we have enough columns
            # Remove bold markers from all cells
            cells = [cell.strip('*') for cell in cells]
            # Check if this is a group header (Task ID ends with .0)
            is_group_header = cells[0].endswith('.0')
            if is_group_header:
                current_phase = cells[2]  # Update current phase
            elif current_phase:  # If we have a current phase, use it
                cells[2] = current_phase  # Ensure task uses the correct phase
            data.append(cells + [is_group_header])
    
    if not data:
        st.error("No valid data rows found in the table")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Task ID', 'Task Name', 'Phase', 'Duration (Weeks)', 
                                   'Start Date', 'End Date', 'Predecessors', 'Deliverables Mapping', 'IsGroupHeader'])
    
    # Convert dates with explicit format
    df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d/%m/%Y', errors='coerce')
    df['End Date'] = pd.to_datetime(df['End Date'], format='%d/%m/%Y', errors='coerce')
    
    # Remove any rows where date parsing failed
    df = df.dropna(subset=['Start Date', 'End Date'])
    
    if df.empty:
        st.error("No valid rows with dates found in the table")
        return None
    
    # Create the format required by create_gantt
    gantt_df = pd.DataFrame({
        'Task': df['Task ID'] + ': ' + df['Task Name'],
        'Start': df['Start Date'],
        'Finish': df['End Date'],
        'Resource': df['Phase'],
        'IsGroupHeader': df['IsGroupHeader']
    })
    
    return gantt_df

# Generate button
if st.button("Generate Gantt Chart"):
    if project_plan:
        with st.spinner("Converting project plan to Gantt chart..."):
            # Convert project plan to markdown table using GPT-4
            markdown_table = convert_to_gantt(project_plan)
            
            if markdown_table:
                # Display the markdown table
                st.markdown("### Generated Gantt Chart Table")
                st.markdown(markdown_table)
                
                try:
                    # Parse markdown table to DataFrame
                    df = parse_markdown_table(markdown_table)
                    
                    # Create color mapping for phases
                    unique_phases = df['Resource'].unique()
                    colors = {
                        'Planning': 'rgb(46, 137, 205)',
                        'Development': 'rgb(114, 44, 121)',
                        'Testing': 'rgb(198, 47, 105)',
                        'Discovery': 'rgb(58, 149, 136)',
                        'Strategy': 'rgb(107, 127, 135)',
                        'Requirements': 'rgb(46, 137, 205)',
                        'Design': 'rgb(114, 44, 121)',
                        'Implementation': 'rgb(198, 47, 105)',
                        'Deployment': 'rgb(58, 149, 136)',
                        'Maintenance': 'rgb(107, 127, 135)'
                    }
                    
                    # Filter colors to only include phases that exist in the data
                    phase_colors = {phase: colors.get(phase, 'rgb(128, 128, 128)') for phase in unique_phases}
                    
                    # Create Gantt chart
                    fig = ff.create_gantt(df, 
                                        index_col='Resource',
                                        show_colorbar=True,
                                        group_tasks=True,
                                        showgrid_x=True,
                                        showgrid_y=True,
                                        colors=phase_colors)
                    
                    # Update layout
                    fig.update_layout(
                        title='Project Gantt Chart',
                        height=800,  # Increased height
                        font=dict(size=12),  # Increased font size
                        showlegend=True,
                        xaxis_title="Timeline",
                        yaxis_title="Tasks",
                        margin=dict(t=100, b=50, l=50, r=50)  # Added margins
                    )
                    
                    # Update y-axis to show phases
                    phase_tasks = []
                    for phase in unique_phases:
                        # Get group header for this phase
                        phase_header = df[df['IsGroupHeader'] & (df['Resource'] == phase)]['Task'].iloc[0]
                        phase_tasks.append(phase_header)
                        # Add tasks for this phase
                        phase_tasks.extend(df[~df['IsGroupHeader'] & (df['Resource'] == phase)]['Task'].tolist())
                    
                    fig.update_yaxes(
                        categoryorder='array',
                        categoryarray=phase_tasks
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating Gantt chart: {str(e)}")
    else:
        st.warning("Please enter a project plan first.")

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