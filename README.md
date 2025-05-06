# Gantt Chart Generator

A Streamlit application for creating and visualizing Gantt charts from project plans.

## Setup

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

1. Activate the Poetry environment:
```bash
poetry shell
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Features

- Convert project plans to Gantt charts
- Edit data in a table view
- Download Gantt charts as HTML
- Interactive visualization
- AI-powered project plan analysis
