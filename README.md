# Ask Syracuse Data

A natural language interface for querying Syracuse Open Data. Ask questions in plain English and get data-driven answers with interactive visualizations.

## Features

- **Natural Language Queries**: Ask questions like "What's the crime rate by neighborhood?" or "Which zip codes have the most violations?"
- **Interactive Web UI**: Modern FastAPI + Tailwind CSS interface with charts and maps
- **Cross-Dataset Analysis**: Join violations with rental properties, vacant properties, and more
- **Auto-Insights**: AI-generated insights explaining what the data means
- **Data Quality Handling**: Robust null value handling with configurable strategies

## Datasets

Four Syracuse Open Data sources (static CSV snapshots):

| Dataset | Description | Key Fields |
|---------|-------------|------------|
| Code Violations | Housing code enforcement (2017-present) | neighborhood, violation type, status |
| Rental Registry | Rental property inspections | zip, validity dates, SBL |
| Vacant Properties | Administratively identified vacancies | neighborhood, registry status |
| Crime Data 2022 | Part 1 offenses with geocoded neighborhoods | crime type, neighborhood, arrest status |

## Architecture

```
User Question → Intent Parser (GPT-4) → Schema Validation → SQL Builder → DuckDB → Results + Charts
```

- **Intent Parser**: Converts natural language to structured JSON intent
- **Schema Validation**: Enforces allowed datasets, fields, and filters
- **SQL Builder**: Generates deterministic DuckDB queries (single-table and cross-dataset joins)
- **Data Quality**: Handles nulls with configurable strategies (label, drop, keep)

## Tech Stack

- **Backend**: FastAPI + DuckDB
- **Frontend**: Tailwind CSS + Plotly.js
- **LLM**: OpenAI GPT-4 (intent parsing only - never touches data)
- **Deployment**: Render-ready (render.yaml included)

## Run Locally

```bash
# Clone and setup
git clone https://github.com/prathyusha1231/Ask-Syracuse-Data.git
cd Ask-Syracuse-Data
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Set OpenAI API key
copy .env.example .env  # then add OPENAI_API_KEY=...

# Download data files to data/raw/ (see Data Sources below)

# Run the web app
python -m uvicorn app:app --reload
# Open http://127.0.0.1:8000
```

## Example Queries

| Question | What it does |
|----------|--------------|
| "How many code violations are there?" | Total count |
| "Which neighborhoods have the most crime?" | Crime counts grouped by neighborhood |
| "Show violations by status" | Breakdown by open/closed/etc |
| "Which zip codes have rental properties with violations?" | Cross-dataset join analysis |

## Data Sources

Download from [Syracuse Open Data](https://data.syr.gov):
- Code_Violations_V2.csv
- Syracuse_Rental_Registry.csv
- Vacant_Properties.csv
- Crime_Data_2022_(Part_1_Offenses).csv

Place in `data/raw/` directory.

## LLM Usage

- **Scope**: LLM converts natural language → JSON intent only
- **Guardrails**: All intents validated against schema; unsupported queries rejected
- **Data Access**: LLM never sees or executes SQL; all computations are deterministic

## Limitations

- Administrative records reflect reporting/enforcement patterns, not ground truth
- Crime data is block-level (addresses generalized for privacy)
- Counts should be normalized for fair neighborhood comparisons
- Outputs are descriptive, not causal - always consider context

## License

MIT
