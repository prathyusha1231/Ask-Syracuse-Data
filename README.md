# Ask Syracuse Data

A natural language interface for querying Syracuse Open Data. Ask questions in plain English and get data-driven answers with interactive visualizations, validation, and bias warnings.

## Features

- **Natural Language Queries**: Ask questions like "How many violations by neighborhood?" or "What's the average property assessment?"
- **16 Datasets**: Housing, public safety, city services, infrastructure, and public health data
- **Interactive Web UI**: Landing page with dataset explorer, charts, maps, and tabbed results
- **Cross-Dataset Analysis**: Join violations with rental properties, vacant properties, crime data, and unfit properties
- **Hybrid Architecture**: Simple queries use deterministic SQL; complex queries use LLM-generated SQL with guardrails
- **Auto-Insights**: AI-generated insights explaining what the data means
- **Validation**: Ground-truth comparison ensures query results match direct pandas calculations
- **Bias Detection**: Automatic warnings for framing, normalization, selection, and context biases
- **Data Citations**: Full source attribution with dataset caveats and limitations

## Datasets

16 Syracuse Open Data sources (static CSV snapshots):

| Category | Dataset | Records | Description |
|----------|---------|---------|-------------|
| **Housing & Property** | Code Violations | ~44K | Housing code enforcement (2017-present) |
| | Vacant Properties | ~1.4K | Administratively identified vacancies |
| | Rental Registry | ~13K | Registered rental property inspections |
| | Unfit Properties | 353 | Properties declared unfit for habitation |
| | Historical Properties | 3,486 | Landmark and National Register eligible properties |
| | Assessment Roll | ~41K | Property assessments and classifications (2026) |
| **Public Safety** | Crime Data 2022 | ~4K | Part 1 offenses with geocoded neighborhoods |
| | Parking Violations | ~197K | Parking tickets issued (2023) |
| **City Services** | SYRCityline Requests | ~116K | 311 service requests |
| | Trash Pickup | ~41K | Collection schedules (2025) |
| | Permit Requests | ~47K | Building permit applications |
| **Infrastructure** | Bike Suitability | 868 | Road bike suitability ratings (2020) |
| | Bike Infrastructure | 59 | Bike lanes, trails, and paths (2023) |
| | Snow Routes | 3,685 | Emergency snow route road segments |
| | Tree Inventory | ~55K | City-managed tree inventory |
| **Public Health** | Lead Testing | ~157 | Elevated lead levels by census tract (2013-2024) |

## Architecture

```
User Question
    |
Intent Parser (detects complexity)
    |-- Path 1 (Simple/Medium) --> Intent JSON --> Schema Validator --> SQL Builder --> DuckDB
    |   Supports: count, count_distinct, avg/min/max/sum, temporal GROUP BY,
    |             HAVING, multi GROUP BY, cross-dataset joins
    |
    +-- Path 2 (Complex) --> LLM SQL Generation --> SQL Validator (guardrails) --> DuckDB
        Supports: window functions, CASE WHEN, subqueries, rankings, percentiles
```

- **Intent Parser**: Converts natural language to structured JSON (GPT-4o-mini or heuristic fallback)
- **Schema Validation**: Enforces allowed datasets, fields, metrics, and filters
- **SQL Builder**: Generates deterministic DuckDB queries (single-table, joins via CTE or LEFT JOIN)
- **SQL Validator**: Guardrails for LLM-generated SQL (read-only, table allowlist, LIMIT 1000)
- **Validation**: Compares results against pandas ground-truth calculations
- **Bias Detection**: Warns about framing, normalization, selection, and context biases

## Tech Stack

- **Backend**: FastAPI + DuckDB (in-memory SQL on pandas DataFrames)
- **Frontend**: Tailwind CSS + Plotly.js (single-page app)
- **LLM**: OpenAI GPT-4o-mini (intent parsing only — never touches data)
- **Deployment**: Hugging Face Spaces (Docker) or Render.com

## Run Locally

```bash
# Clone and setup
git clone https://github.com/prathyusha1231/Ask-Syracuse-Data.git
cd Ask-Syracuse-Data
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Optional: Set OpenAI API key for AI features (app works without it)
# Create .env file with: OPENAI_API_KEY=sk-...

# Download data files to data/raw/ (see Data Sources below)

# Run the web app
python -m uvicorn app:app --reload
# Open http://127.0.0.1:8000
```

## Example Queries

| Category | Question | What it does |
|----------|----------|--------------|
| Housing | "How many code violations are there?" | Total count |
| Housing | "Violations by neighborhood since 2020" | Filtered, grouped count |
| Housing | "Average days to comply by neighborhood" | Computed column metric |
| Housing | "Neighborhoods with more than 100 violations" | HAVING threshold |
| Safety | "Crime by type" | Crime breakdown |
| Safety | "Parking violations by zip code" | Parking tickets grouped |
| Services | "Service requests by category" | 311 request breakdown |
| Services | "Trash pickup by collection day" | Schedule breakdown |
| Property | "Average property assessment by class" | Assessment metrics |
| Property | "Historical properties by NR eligibility" | Landmark data |
| Infra | "Bike infrastructure by type" | Lane/trail/path counts |
| Infra | "How many miles of bike lanes?" | Sum metric |
| Trees | "Most common tree species" | Species inventory |
| Health | "Lead testing by census tract" | Public health data |
| Joins | "Rental properties with violations by zip" | Cross-dataset join |
| Joins | "Compare crime and violations by neighborhood" | Crime + violations |

## Data Sources

Download from [Syracuse Open Data](https://data.syr.gov) and place in `data/raw/`:

- `Code_Violations_V2.csv`
- `Syracuse_Rental_Registry.csv`
- `Vacant_Properties.csv`
- `Crime_Data_2022_(Part_1_Offenses).csv`
- `Unfit_Properties.csv`
- `Trash_Pickup_2025.csv`
- `Historical_Properties.csv`
- `Assessment_Roll_2026.csv`
- `SYRCityline_Requests.csv`
- `Emergency_Snow_Routes.csv`
- `Bike_Suitability_2020.csv`
- `Bike_Infrastructure_2023.csv`
- `Parking_Violations_2023.csv`
- `Permit_Requests.csv`
- `Tree_Inventory.csv`
- `Lead_Testing_2013_2019.xlsx`
- `Lead_Testing_2020_2024.xlsx`

## LLM Usage

- **Scope**: LLM converts natural language to JSON intent only (Path 1) or generates SQL with guardrails (Path 2)
- **Heuristic Fallback**: Common queries work without an API key via keyword-based intent parsing
- **Guardrails**: All intents validated against schema; LLM SQL restricted to read-only, allowed tables only, LIMIT 1000
- **Data Access**: The LLM never sees raw data — all computations happen in DuckDB

## Validation & Bias Detection

**Validation (validation.py)**
- Compares query results against ground-truth pandas calculations
- Sanity checks for outliers, null groups, and suspicious counts
- Validates join results against direct merge operations

**Bias Detection (bias_detection.py)**
- **Framing**: Warns about leading language ("most dangerous", "worst")
- **Normalization**: Alerts when raw counts need population context
- **Selection**: Notes dataset-specific collection biases
- **Context**: Identifies missing analytical context
- **Uncertainty**: Flags sources of uncertainty in results

View these in the **Validation** and **Sources** tabs in the web UI.

## Testing

```bash
# Benchmark tests (12 tests, no LLM needed)
python -m tests.eval_benchmarks

# With LLM
python -m tests.eval_benchmarks --llm

# Comprehensive tests (30 questions)
python -m tests.test_app_comprehensive
```

## Deploy

**Hugging Face Spaces (Docker):**
1. Create a new Space with Docker SDK (Blank template)
2. Copy project files + `data/raw/` into the Space repo
3. Push — the included `Dockerfile` handles the build
4. Set `OPENAI_API_KEY` as a secret in Space Settings (optional)

**Render.com:**
- `render.yaml` is included — connect your GitHub repo and set `OPENAI_API_KEY` in the dashboard

## Limitations

- Administrative records reflect reporting/enforcement patterns, not ground truth
- Crime data covers 2022 only (Part 1 offenses); addresses generalized to block level
- Counts should be normalized for fair neighborhood comparisons
- Assessed values may differ from market values
- Lead testing data is census-tract level (research use)
- Static CSV snapshots — not live data

## License

MIT
