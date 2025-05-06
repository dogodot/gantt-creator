import pandas as pd
import plotly.figure_factory as ff
from datetime import datetime

# Define the data
data = [
    # Phase 1: Initiation & Discovery
    dict(Task="Phase 1: Initiation & Discovery", Start='2025-05-19', Finish='2025-05-30', Resource='Discovery'),
    dict(Task="1.1 Project Kick-off & Governance Setup", Start='2025-05-19', Finish='2025-05-21', Resource='Discovery'),
    dict(Task="1.2 Request & Review Plenitude Documents", Start='2025-05-19', Finish='2025-05-28', Resource='Discovery'),
    dict(Task="1.3 Stakeholder Mapping & Engagement Planning", Start='2025-05-22', Finish='2025-05-28', Resource='Discovery'),
    dict(Task="1.4 Initial Workshops (AI Leads/Sponsor)", Start='2025-05-26', Finish='2025-05-30', Resource='Discovery'),
    
    # Phase 2: Strategy Definition & Alignment
    dict(Task="Phase 2: Strategy Definition & Alignment", Start='2025-06-02', Finish='2025-06-13', Resource='Strategy Definition'),
    dict(Task="2.1 Leadership Workshops", Start='2025-06-02', Finish='2025-06-06', Resource='Strategy Definition'),
    dict(Task="2.2 Analysis & Workshops", Start='2025-06-02', Finish='2025-06-06', Resource='Strategy Definition'),
    dict(Task="2.3 Policy Review & Update Recommendations", Start='2025-06-09', Finish='2025-06-13', Resource='Strategy Definition'),
    
    # Phase 3: Use Case Assessment & Prioritisation
    dict(Task="Phase 3: Use Case Assessment & Prioritisation", Start='2025-06-09', Finish='2025-06-25', Resource='Use Case Definition'),
    dict(Task="3.1 Cross-functional Workshops", Start='2025-06-09', Finish='2025-06-18', Resource='Use Case Definition'),
    dict(Task="3.2 Vendor Input & Sector Analysis", Start='2025-06-16', Finish='2025-06-18', Resource='Use Case Definition'),
    dict(Task="3.3 Use Case Analysis & Prioritisation", Start='2025-06-19', Finish='2025-06-25', Resource='Use Case Definition'),
    dict(Task="3.4 Draft ROI Framework Concept", Start='2025-06-23', Finish='2025-06-25', Resource='Use Case Definition'),
    
    # Phase 4: Enablement & Planning
    dict(Task="Phase 4: Enablement & Planning", Start='2025-06-26', Finish='2025-07-11', Resource='Planning & Requirements'),
    dict(Task="4.1 IT Requirements Input Definition", Start='2025-06-26', Finish='2025-07-02', Resource='Planning & Requirements'),
    dict(Task="4.2 Finalise ROI & Adoption Tracking", Start='2025-07-03', Finish='2025-07-09', Resource='Planning & Requirements'),
    dict(Task="4.3 Conduct Training Needs Analysis", Start='2025-06-26', Finish='2025-07-07', Resource='Planning & Requirements'),
    dict(Task="4.4 Analyse TNA Findings", Start='2025-07-07', Finish='2025-07-11', Resource='Planning & Requirements'),
    
    # Phase 5: Consolidation & Handover
    dict(Task="Phase 5: Consolidation & Handover", Start='2025-07-14', Finish='2025-07-25', Resource='Finalisation'),
    dict(Task="5.2 Draft & Consolidate Final Report", Start='2025-07-14', Finish='2025-07-18', Resource='Finalisation'),
    dict(Task="5.3 Incorporate Plenitude Feedback", Start='2025-07-21', Finish='2025-07-23', Resource='Finalisation'),
    dict(Task="5.4 Final Presentation & Handover", Start='2025-07-24', Finish='2025-07-25', Resource='Finalisation'),
    dict(Task="5.5 Project Closure", Start='2025-07-25', Finish='2025-07-25', Resource='Finalisation'),
]

# Create DataFrame
df = pd.DataFrame(data)

# Define colors for different phases
colors = {
    'Discovery': 'rgb(46, 137, 205)',
    'Strategy Definition': 'rgb(114, 44, 121)',
    'Use Case Definition': 'rgb(198, 47, 105)',
    'Planning & Requirements': 'rgb(58, 149, 136)',
    'Finalisation': 'rgb(107, 127, 135)'
}

# Create the Gantt chart
fig = ff.create_gantt(df, 
                      colors=colors,
                      index_col='Resource',
                      show_colorbar=True,
                      group_tasks=True,
                      showgrid_x=True,
                      showgrid_y=True,
                      height=800,
                      title='Project Timeline Gantt Chart')

# Update layout
fig.update_layout(
    title_x=0.5,
    title_font_size=24,
    font=dict(size=12),
    xaxis_title="Date",
    yaxis_title="Tasks",
    showlegend=True
)

# Save the chart as an HTML file
fig.write_html("project_gantt_chart.html")

print("Gantt chart has been generated and saved as 'project_gantt_chart.html'") 