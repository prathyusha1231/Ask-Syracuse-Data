# Technical Documentation - Ask Syracuse Data

## Architecture Overview

Ask Syracuse Data is a full-stack web application that translates natural language questions into SQL queries against 16 Syracuse Open Data CSV datasets. The system uses a hybrid architecture with two query paths:

```
User Question (plain English)
       |
  Intent Parser (LLM or heuristic)
       |
  Complexity Triage
       |
       +-- Path 1 (Simple/Medium) -----> Intent JSON
       |                                    |
       |                              Schema Validator
       |                                    |
       |                              SQL Builder (deterministic)
       |                                    |
       |                              DuckDB (in-memory)
       |                                    |
       |                              Validation + Bias Detection
       |
       +-- Path 2 (Complex) ----------> LLM SQL Generation
       |                                    |
       |                              SQL Validator (guardrails)
       |                                    |
       |                              DuckDB (in-memory)
       |
       +-- Path 3 (Cross-Dataset) ---> Join Intent JSON
                                            |
                                       Schema Validator
                                            |
                                       SQL Builder (CTE/LEFT JOIN)
                                            |
                                       DuckDB (in-memory)
                                            |
                                       Validation + Bias Detection
```

### Key Design Decisions

1. **DuckDB over SQLite/Postgres**: DuckDB runs in-memory on pandas DataFrames with zero setup. No database server needed. Ideal for read-only analytics on static CSVs.

2. **Hybrid LLM usage**: The LLM is used only for intent parsing (Path 1) or SQL generation (Path 2) - it never sees raw data. This minimizes cost, latency, and hallucination risk.

3. **Heuristic fallback**: A keyword-based parser handles common queries without any API key, making the app functional offline.

4. **Deterministic SQL for simple queries**: Path 1 generates SQL programmatically from validated intent JSON. This is faster, cheaper, and fully testable compared to LLM-generated SQL.

5. **Guardrails on LLM SQL**: Path 2 SQL passes through `sql_validator.py` which enforces read-only operations, a table allowlist, and a LIMIT 1000 cap before execution.

---

## Development Setup

### Prerequisites

- Python 3.10+
- pip
- (Optional) OpenAI API key for AI-powered features

### Installation

```bash
# Clone the repository
git clone https://github.com/prathyusha1231/Ask-Syracuse-Data.git
cd Ask-Syracuse-Data

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Place CSV/XLSX files in data/raw/ (see Data Sources in README.md)
mkdir -p data/raw

# (Optional) Create .env for AI features
echo "OPENAI_API_KEY=sk-..." > .env

# Start development server
python -m uvicorn app:app --reload
# Open http://127.0.0.1:8000
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework (async, Pydantic models) |
| `uvicorn` | ASGI server |
| `duckdb` | In-memory SQL engine on DataFrames |
| `pandas` / `numpy` | Data loading, transformation, validation |
| `openai` | GPT-4o-mini for intent parsing and SQL generation |
| `jinja2` | HTML template rendering |
| `slowapi` | Rate limiting (20 queries/min, 30 feedback/min) |
| `openpyxl` | Excel file support (lead testing data) |
| `pytest` | Test framework |
| `plotly` / `matplotlib` / `seaborn` | Visualization (notebooks) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | Enables LLM intent parsing, AI insights, and Path 2 SQL generation. Without it, heuristic parser handles common queries. |
| `PORT` | No | Server port (default: 8000). Auto-set on Render/HF Spaces. |

---

## Code Organization

```
Ask-Syracuse-Data/
|-- app.py                          # FastAPI entry point (1,163 lines)
|   |-- SecurityHeadersMiddleware   # CSP, X-Frame-Options, etc.
|   |-- QueryRequest / QueryResponse # Pydantic models
|   |-- POST /api/query             # Main query endpoint (rate-limited)
|   |-- POST /api/feedback          # User feedback collection
|   |-- GET  /api/feedback/stats    # Feedback statistics
|   |-- GET  /api/health            # Health check
|   |-- GET  /                      # Serves index.html
|   |-- generate_insights()         # GPT-4o-mini data insights
|   |-- generate_map_data()         # ZIP/neighborhood/point/route line maps
|   +-- DATA_CITATIONS              # Source attribution for 16 datasets
|
|-- pipeline/                       # Core query pipeline
|   |-- main.py                     # Orchestrator (466 lines)
|   |   |-- run_query()             # Entry point: parse -> validate -> SQL -> execute -> validate
|   |   |-- _run_single_query()     # Path 1: single-dataset deterministic
|   |   |-- _run_join_query()       # Path 3: cross-dataset joins
|   |   |-- _run_advanced_query()   # Path 2: LLM-generated SQL
|   |   +-- _execute_sql_with_timeout()  # 30s timeout guard
|   |
|   |-- schema.py                   # Dataset configurations
|   |   |-- DATASETS                # 16 dataset configs (table, columns, metrics, filters)
|   |   |-- ALLOWED_JOINS           # 5 cross-dataset join configs
|   |   |-- validate_intent()       # Single-dataset intent validation
|   |   +-- validate_join_intent()  # Join intent validation
|   |
|   |-- sql_builder.py              # Deterministic SQL generation
|   |   |-- build_select_sql()      # Single-table: SELECT, WHERE, GROUP BY, HAVING, LIMIT
|   |   +-- build_join_sql()        # Cross-dataset: CTEs for ZIP/neighborhood, LEFT JOIN for SBL
|   |
|   |-- sql_validator.py            # LLM SQL guardrails
|   |   |-- validate_sql()          # Read-only check, table allowlist, LIMIT injection
|   |   |-- ALLOWED_TABLES          # 16 table names
|   |   +-- SQLValidationError      # Custom exception
|   |
|   |-- data_utils.py               # CSV/XLSX loaders + data cleaning
|   |   |-- load_*()                # 16 loader functions (one per dataset)
|   |   |-- _clean_columns()        # Lowercase, strip column names
|   |   |-- _apply_null_handling()  # Replace nulls with descriptive labels
|   |   |-- SYRACUSE_ZIP_CENTROIDS  # 12 ZIP code coordinates
|   |   +-- DATA_DIR                # Path to data/raw/
|   |
|   |-- validation.py               # Ground-truth validation
|   |   |-- validate_count_result() # Compare SQL count vs pandas count
|   |   |-- validate_join_result()  # Compare join vs pandas merge
|   |   |-- sanity_check_result()   # Outlier, null group, suspicious count checks
|   |   +-- combine_validations()   # Merge multiple validation results
|   |
|   |-- bias_detection.py           # Bias warning generation
|   |   |-- run_all_bias_checks()   # Returns BiasResult with 5 check types
|   |   |-- check_framing_bias()    # Leading language detection
|   |   |-- check_normalization()   # Raw counts needing population context
|   |   |-- check_selection_bias()  # Dataset collection patterns
|   |   |-- check_context_bias()    # Missing analytical context
|   |   +-- check_uncertainty()     # Source uncertainty flags
|   |
|   +-- data_quality.py             # Data quality auditing
|
|-- llm/                            # LLM integration
|   |-- intent_parser.py            # NL -> structured intent
|   |   |-- parse_intent()          # Entry: tries heuristic, falls back to LLM
|   |   |-- _heuristic_parse()      # Keyword-based parsing (no API key needed)
|   |   |-- _needs_advanced_sql()   # Complexity routing to Path 2
|   |   +-- _is_join_query()        # Cross-dataset detection
|   |
|   |-- openai_client.py            # OpenAI API wrappers
|   |   |-- make_openai_intent_llm()# Returns callable for intent parsing
|   |   |-- make_openai_sql_llm()   # Returns callable for SQL generation
|   |   |-- _call_with_retry()      # Exponential backoff (rate limit, timeout, server errors)
|   |   +-- load_api_key()          # .env loader
|   |
|   +-- prompt_templates.py         # 3 prompt templates
|       |-- NL_TO_INTENT_PROMPT     # Single-dataset intent parsing
|       |-- NL_TO_JOIN_INTENT_PROMPT# Cross-dataset join parsing
|       +-- NL_TO_SQL_PROMPT        # Full SQL generation (all 16 table schemas)
|
|-- tests/                          # Test suites
|   |-- conftest.py                 # Auto-disables LLM (FORCE_LLM_TESTS=1 to override)
|   |-- test_pipeline.py            # 73 pytest tests (10 test classes)
|   |-- eval_benchmarks.py          # 13 benchmark tests
|   |-- test_app_comprehensive.py   # 30 comprehensive tests
|   +-- test_all_datasets.py        # 37 dataset tests (requires running server)
|
|-- templates/
|   +-- index.html                  # Single-page app (Tailwind CSS + Plotly.js)
|
|-- notebooks/
|   +-- 03_full_eda.ipynb           # Exploratory data analysis
|
|-- scripts/
|   +-- fix_unknown_neighborhoods.py# Backfill Unknown neighborhoods in crime data
|
|-- docs/                           # Architecture plans, week reports
|-- data/raw/                       # CSV/XLSX files (gitignored)
|-- Dockerfile                      # HF Spaces deployment
|-- render.yaml                     # Render.com deployment
+-- requirements.txt                # Python dependencies
```

---

## API Documentation

### `POST /api/query`

Process a natural language question about Syracuse data.

**Rate limit**: 20 requests/minute per IP

**Request body**:
```json
{
  "question": "How many violations by neighborhood?"
}
```
- `question` (string, required): 1-500 characters

**Response** (`QueryResponse`):
```json
{
  "success": true,
  "query_id": "uuid-string",
  "description": "Code Violations grouped by Neighborhood. Found 32 groups.",
  "columns": ["Neighborhood", "Total Count"],
  "data": [{"Neighborhood": "Northside", "Total Count": "12,345"}, ...],
  "chart_data": {"type": "bar", "labels": [...], "values": [...]},
  "map_data": {"type": "neighborhood_bubble", "lats": [...], ...},
  "insights": "AI-generated analysis of the results...",
  "sql": "SELECT neighborhood, COUNT(*) AS count FROM violations GROUP BY neighborhood",
  "metadata": {"query_type": "single", "dataset": "violations", ...},
  "limitations": "Static CSV snapshots only...",
  "validation": {"passed": true, "warnings": [], "errors": [], "ground_truth": {}},
  "bias_warnings": [{"type": "normalization", "message": "..."}],
  "citations": [{"name": "Code Violations V2", "source": "Syracuse Open Data Portal", ...}],
  "clarification": null,
  "error": null
}
```

**Chart types**: `bar`, `line`, `grouped_bar` (auto-selected based on query type)

**Map types**: `neighborhood_bubble`, `zip_bubble`, `point`, `line` (auto-selected based on group-by column and dataset; `line` type used for bike infrastructure and snow route maps)

### `POST /api/feedback`

Submit user feedback on a query result.

**Rate limit**: 30 requests/minute per IP

**Request body**:
```json
{
  "query_id": "uuid-string",
  "question": "How many violations?",
  "rating": "up",
  "comment": "Accurate result",
  "sql": "SELECT COUNT(*) ...",
  "dataset": "violations"
}
```
- `rating` (string, required): `"up"` or `"down"`

### `GET /api/feedback/stats`

Returns feedback totals and 10 most recent entries.

### `GET /api/health`

Returns `{"status": "healthy", "service": "Ask Syracuse Data"}`.

---

## Testing Instructions

### Test Suites

| Suite | Tests | LLM Required | Command |
|-------|-------|-------------|---------|
| **Pipeline (pytest)** | 73 | No (auto-disabled) | `pytest tests/test_pipeline.py -v` |
| **Benchmarks** | 13 | No (heuristic) | `python -m tests.eval_benchmarks` |
| **Benchmarks (LLM)** | 13 | Yes | `python -m tests.eval_benchmarks --llm` |
| **Comprehensive** | 30 | Optional | `python -m tests.test_app_comprehensive` |
| **All Datasets** | 37 | Optional | `python -m tests.test_all_datasets` |

### Pytest Test Classes (73 tests in `tests/test_pipeline.py`)

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestIntentParsing` | 15 | Heuristic parsing for all datasets, joins, filters, metrics |
| `TestMetricQueries` | 4 | count, parking fines, cityline categories |
| `TestTemporalQueries` | 3 | Violations/crime by year |
| `TestHavingQueries` | 3 | Threshold filtering ("more than N", "over N") |
| `TestFilterQueries` | 3 | Year equals, year since, ground-truth validation |
| `TestJoinQueries` | 3 | ZIP and SBL cross-dataset joins |
| `TestErrorHandling` | 6 | Empty, nonsense, long, XSS, SQL injection inputs |
| `TestSQLValidator` | 13 | DROP/INSERT/DELETE, semicolons, unknown tables, LIMIT enforcement |
| `TestSchemaValidation` | 5 | Valid/invalid datasets, metrics, normalization |
| `TestGroundTruth` | 4 | Total counts and grouped counts vs pandas |
| `TestAllDatasets` | 14 | Parametrized smoke test for all parseable datasets |

### Running Tests

```bash
# Run all pytest tests
pytest tests/test_pipeline.py -v

# Run a specific test class
pytest tests/test_pipeline.py::TestSQLValidator -v

# Force LLM usage in tests (requires OPENAI_API_KEY)
FORCE_LLM_TESTS=1 pytest tests/test_pipeline.py -v

# Benchmark tests with JSON output
python -m tests.eval_benchmarks --json

# Dataset tests (start server first)
python -m uvicorn app:app &
python -m tests.test_all_datasets
```

### Test Configuration

`tests/conftest.py` auto-disables LLM by unsetting `OPENAI_API_KEY` unless `FORCE_LLM_TESTS=1` is set. This ensures tests are deterministic and free by default.

---

## Deployment Guide

### Hugging Face Spaces (Docker)

1. Create a new Space on huggingface.co with **Docker SDK** (Blank template)
2. Copy project files + `data/raw/` into the Space repo
3. Push - the included `Dockerfile` handles the build:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY pipeline/ ./pipeline/
COPY llm/ ./llm/
COPY templates/ ./templates/
COPY data/raw/ ./data/raw/
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

4. Set `OPENAI_API_KEY` as a secret in Space Settings (optional)

### Render.com

The included `render.yaml` configures a web service:

```yaml
services:
  - type: web
    name: ask-syracuse-data
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
      - key: OPENAI_API_KEY
        sync: false
```

1. Connect your GitHub repo in the Render dashboard
2. Set `OPENAI_API_KEY` manually in the environment variables
3. Deploy

### Local Production

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Security

### Implemented Safeguards

| Category | Measure | Location |
|----------|---------|----------|
| **Input validation** | 500-char max question length | `app.py` (Pydantic) |
| **Rate limiting** | 20 queries/min, 30 feedback/min | `app.py` (slowapi) |
| **SQL injection** | Semicolon rejection, read-only enforcement | `sql_validator.py` |
| **XSS prevention** | `escapeHtml()` on all dynamic content | `index.html` |
| **CSP headers** | Script/style/img/connect-src allowlists | `SecurityHeadersMiddleware` |
| **Frame protection** | `X-Frame-Options: DENY` | `SecurityHeadersMiddleware` |
| **SQL timeout** | 30s execution limit via ThreadPoolExecutor | `main.py` |
| **LLM SQL guardrails** | Table allowlist (16 tables), LIMIT 1000 cap | `sql_validator.py` |
| **Retry logic** | Exponential backoff on OpenAI errors | `openai_client.py` |
| **Cache TTL** | 1-hour DataFrame cache invalidation | `main.py` |
| **Frontend timeout** | 60s AbortController timeout | `index.html` |
| **Accessibility** | ARIA labels on input, submit, feedback buttons | `index.html` |
| **SRI hashes** | Plotly.js integrity hash | `index.html` |

### Data Privacy

- Crime addresses are generalized to block level (source data)
- Lead testing data is census-tract level only
- No personally identifiable information is stored
- Feedback database stores questions and ratings only (no IP addresses)
- The LLM never sees raw data - only intent or SQL generation prompts
