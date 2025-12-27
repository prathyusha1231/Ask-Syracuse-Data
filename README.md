# 🚧 Work in Progress
This project is actively being developed. Features, documentation, and structure may change as improvements are made.

# Ask Syracuse Data

Governed, deterministic analysis of Syracuse housing and public safety CSV snapshots. Natural language questions are translated into structured intents (with tight guardrails) and executed via DuckDB SQL. No chatbot behavior; no model-generated answers.

## What it is
- Static, reproducible pipeline over four Syracuse Open Data CSVs (code violations, rental registry, vacant properties, 2022 Part 1 crime).
- Deterministic loaders (`data_utils.py`), strict intent validation and aliases (`schema.py`), intent parsing (GPT or heuristic), SQL builder (`sql_builder.py`), and DuckDB execution (in-memory).
- CLI entry point (`main.py`) and an optional test-only Streamlit UI (`ui_streamlit.py`) that calls the same backend.

## What it does today
- Load/clean CSVs from `data/raw/` with normalized columns and parsed dates.
- Guardrails: allowed datasets/group-bys/filters, dataset/group_by aliases, and validation of all intents.
- SQL generation: pure Python builder creates DuckDB `SELECT` queries (no joins across datasets, no mutations).
- Execution: DuckDB runs generated SQL; results, SQL, metadata, and limitations are returned to CLI/UI.

## What it does **not** do
- No predictive or causal modeling; descriptive counts only.
- No live data ingestion or external APIs beyond optional GPT for intent parsing.
- No LLM access to data or SQL execution; LLM outputs intent JSON only.

## Data sources (static snapshots)
- `data/raw/Code_Violations_V2.csv` — 2017–present code enforcement violations (addresses, violation types, status, neighborhood, lat/long).
- `data/raw/Syracuse_Rental_Registry.csv` — rental registry inspections/validity (SBL, zip, validity dates, lat/long).
- `data/raw/Vacant_Properties.csv` — administratively identified vacant properties (registry status, neighborhood, lat/long).
- `data/raw/Crime_Data_2022_(Part_1_Offenses).csv` — reported Part 1 offenses in 2022 (offense type, date, block-level address).

## LLM usage and limits
- Scope: LLM converts NL -> JSON intent only (response_format JSON). Temperature low.
- Guardrails: intents validated against `schema.py`; unsupported datasets/fields are rejected.
- Data access: LLM never touches data; computations are deterministic DuckDB queries.

## Run locally
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# optional: set GPT key for intent parsing
copy .env.example .env  # then set OPENAI_API_KEY=...
python main.py "Which neighborhoods have the most code violations?"
```
Streamlit test UI (optional):
```
streamlit run ui_streamlit.py
```

## Recent tests (2025-12-27)
- "How many code violations are there?" -> 137,663 (SQL: SELECT count(*) FROM violations)
- "How many vacant properties are there?" -> 1,651 (SQL: SELECT count(*) FROM vacant_properties)
- "How many crimes were reported in 2022?" -> 5,642 (SQL with year filter on dateend)
- "Which neighborhoods have the most code violations?" -> counts by neighborhood
- "Which zip codes have the most vacant properties?" -> counts by zip (zip_code alias supported)
- "What types of crimes were most common in 2022?" -> counts by code_defined
- Rejected by guardrails (expected): cross-dataset questions (e.g., crime vs rental, vacant vs violations) and unsupported group_bys

## Roadmap
- [ ] Improve SQL intent routing accuracy
- [ ] Add more example NL → SQL queries
- [ ] Refactor schema handling
- [ ] Add evaluation benchmarks

## Limitations / ethical framing
- Administrative records are not ground truth; reporting/enforcement vary by area.
- Crime is block-level and reported incidents only.
- Counts should be normalized for neighborhood comparisons; avoid stigmatizing language.
- Outputs are descriptive, not causal; always present limitations with results.
