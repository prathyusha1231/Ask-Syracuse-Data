# Methodology - Ask Syracuse Data

## Overview

This document describes the analytical approach, data processing methods, LLM integration strategy, validation techniques, and known limitations of the Ask Syracuse Data project.

---

## 1. Data Processing

### 1.1 Data Acquisition

All datasets are static CSV/XLSX snapshots downloaded from the Syracuse Open Data Portal (data.syr.gov). Data files are stored in `data/raw/` (gitignored) and loaded at runtime by dedicated loader functions in `pipeline/data_utils.py`.

| Dataset | Source Files | Records | Format |
|---------|-------------|---------|--------|
| Code Violations | `Code_Violations_V2.csv` | ~138,000 | CSV |
| Crime Data | 6 CSVs (2022-2025, Part 1 & 2) | ~32,840 | CSV |
| Rental Registry | `Syracuse_Rental_Registry.csv` | ~13,000 | CSV |
| Vacant Properties | `Vacant_Properties.csv` | ~1,700 | CSV |
| Unfit Properties | `Unfit_Properties.csv` | 353 | CSV |
| Trash Pickup | `Trash_Pickup_2025.csv` | ~41,000 | CSV |
| Historical Properties | `Historical_Properties.csv` | 3,486 | CSV |
| Assessment Roll | `Assessment_Roll_2026.csv` | ~41,000 | CSV |
| Cityline Requests | `SYRCityline_Requests.csv` | ~116,000 | CSV |
| Snow Routes | `Emergency_Snow_Routes.csv` | 3,685 | CSV |
| Bike Suitability | `Bike_Suitability_2020.csv` | 868 | CSV |
| Bike Infrastructure | `Bike_Infrastructure_2023.csv` | 59 | CSV |
| Parking Violations | `Parking_Violations_2023.csv` | ~197,000 | CSV |
| Permit Requests | `Permit_Requests.csv` | ~47,000 | CSV |
| Tree Inventory | `Tree_Inventory.csv` | ~55,000 | CSV |
| Lead Testing | 2 XLSX files (2013-2024) | 1,185 | Excel |

### 1.2 Data Cleaning Pipeline

Each dataset has a dedicated loader function (`load_*()`) in `data_utils.py`. The common cleaning steps are:

1. **Column normalization**: All column names are lowercased and stripped of whitespace via `_clean_columns()`.

2. **Type coercion**: Specific columns are cast to appropriate types:
   - `amount` (parking violations): `pd.to_numeric(errors="coerce")` to handle non-numeric entries
   - `year` columns: extracted from date fields and cast to nullable `Int64`
   - Lead testing year columns: converted from integer Excel headers to strings before cleaning

3. **Null handling**: 12 datasets use a "label" strategy where null values are replaced with descriptive text (e.g., "Not Recorded", "Not Scheduled") so they appear in GROUP BY results instead of being silently dropped. This is configured per-dataset in `NULL_STRATEGIES` and applied via `_apply_null_handling()`.

4. **SBL normalization**: Property identifier fields (SBL) are uppercased and stripped for reliable cross-dataset joins.

5. **Date parsing**: Crime data requires re-parsing dates with `utc=True` after concatenating CSVs with mixed date formats.

6. **Pre-2022 filtering**: Stray crime records from before 2022 (found in the 2023 Part 2 file) are filtered out.

### 1.3 Derived Fields

Several fields are computed during data loading:

| Dataset | Derived Field | Method |
|---------|--------------|--------|
| Violations | `days_to_comply`, `days_open`, `cert_duration_days` | Date arithmetic between compliance/open/cert dates |
| Crime | `zip` | Nearest-centroid assignment from lat/long to 12 Syracuse ZIP codes |
| Crime | `neighborhood` | Geocoded from addresses using geopy/Nominatim with nearest-centroid fallback |
| Crime | `crime_part` | Tagged "Part 1" or "Part 2" based on source CSV filename |
| Assessment Roll | `zip` | Regex extraction `r'(\d{5})'` from `property_city` field |
| Cityline | `minutes_to_close` | Time difference between open and close timestamps |
| Parking | `amount` | Cleaned to numeric from mixed string/numeric source |

### 1.4 Cross-Dataset Joins

Five cross-dataset join configurations are supported:

| Primary | Secondary | Join Types | Join Key |
|---------|-----------|-----------|----------|
| Violations | Rental Registry | SBL | Property identifier |
| Violations | Vacant Properties | SBL, neighborhood | Property ID or neighborhood |
| Rental Registry | Vacant Properties | SBL | Property identifier |
| Crime | Violations | ZIP, neighborhood | Geographic area |
| Crime | Vacant Properties | ZIP, neighborhood | Geographic area |

Join SQL uses CTEs for ZIP and neighborhood aggregation joins (avoids cross-join explosion) and LEFT JOIN for SBL-based property-level joins. An estimated-rows guard (>10M threshold) prevents accidental cross-joins that could exhaust memory.

Neighborhood names are normalized across datasets via an alias mapping to handle inconsistent naming (e.g., "Tipp Hill" vs "Tipperary Hill").

---

## 2. Analytical Approach

### 2.1 Query Processing Pipeline

The system converts natural language questions to SQL through a multi-step pipeline:

**Step 1 - Intent Parsing**: The user's question is parsed into a structured intent JSON:
```json
{
  "dataset": "violations",
  "metric": "count",
  "group_by": ["neighborhood"],
  "filters": {"year": {"op": ">=", "value": 2020}},
  "having": {"op": ">", "value": 100}
}
```

Two parsers are available:
- **Heuristic parser** (`_heuristic_parse()`): Keyword-based, no API key needed. Handles common patterns like "violations by neighborhood", "crime by year", "average parking fine".
- **LLM parser** (GPT-4o-mini, temperature 0): Used when available. More flexible with natural language variation.

**Step 2 - Complexity Routing**: The `_needs_advanced_sql()` function detects queries requiring window functions, rankings, percentiles, rolling averages, or year-over-year comparisons. These are routed to Path 2 (LLM SQL generation).

**Step 3 - Schema Validation**: `schema.validate_intent()` checks the intent against allowed datasets, metrics, columns, filters, and group-by fields. It normalizes `group_by` to a list and filters to `{op, value}` format.

**Step 4 - SQL Generation**: Either deterministic (`sql_builder.py`) or LLM-generated (`openai_client.py`) SQL is produced.

**Step 5 - Execution**: SQL runs against DuckDB in-memory with a 30-second timeout.

**Step 6 - Validation**: Results are compared against pandas ground-truth calculations.

**Step 7 - Bias Detection**: Warnings are generated for potential biases in interpretation.

### 2.2 Supported Query Types

| Category | Examples | SQL Pattern |
|----------|----------|-------------|
| Simple count | "How many violations?" | `SELECT COUNT(*)` |
| Count distinct | "How many unique properties?" | `SELECT COUNT(DISTINCT sbl)` |
| Aggregation | "Average days to comply" | `SELECT AVG(days_to_comply)` |
| Grouped count | "Violations by neighborhood" | `GROUP BY neighborhood` |
| Multi group | "Crime by neighborhood and year" | `GROUP BY neighborhood, year` |
| Temporal | "Violations by month" | `GROUP BY EXTRACT(MONTH FROM date)` |
| Filtered | "Violations since 2020" | `WHERE year >= 2020` |
| HAVING | "Neighborhoods with >100 violations" | `HAVING COUNT(*) > 100` |
| Cross-dataset | "Rentals with violations by ZIP" | CTE-based join |
| Complex | "Rank neighborhoods by crime" | LLM-generated window function |

### 2.3 Statistical Methods

The system primarily performs **descriptive statistics**:
- Counts and frequencies (COUNT, COUNT DISTINCT)
- Central tendency (AVG)
- Range (MIN, MAX)
- Totals (SUM)
- Grouped distributions (GROUP BY with optional HAVING)

No inferential statistics (hypothesis tests, confidence intervals, regression) are performed. The system explicitly warns users when questions require subjective judgment ("safest", "best") rather than data queries.

---

## 3. LLM Usage and Validation

### 3.1 LLM Integration Points

| Component | Model | Purpose | Temperature |
|-----------|-------|---------|-------------|
| Intent parsing | GPT-4o-mini | NL -> structured JSON | 0 |
| SQL generation | GPT-4o-mini | NL -> DuckDB SQL (Path 2) | 0 |
| Insights | GPT-4o-mini | Data -> narrative analysis | 0.3 |

### 3.2 Prompt Engineering

Three prompt templates guide LLM behavior:

1. **`NL_TO_INTENT_PROMPT`**: Instructs the model to output a JSON object with dataset, metric, group_by, filters, having, and limit fields. Includes explicit rules:
   - "most common X" -> use count, not count_distinct
   - Don't add HAVING or LIMIT for superlative queries ("most", "top")
   - "arrest" queries -> add filter, not group-by
   - Few-shot examples for common patterns

2. **`NL_TO_JOIN_INTENT_PROMPT`**: Similar structure for cross-dataset queries. Specifies allowed join pairs and types.

3. **`NL_TO_SQL_PROMPT`**: Full schema dump for all 16 tables with column names and types. Includes DuckDB syntax rules, safety constraints (read-only, LIMIT), and examples.

### 3.3 Prompt Iteration History

Key prompt refinements made during development:
- Added rules against spurious HAVING on superlative queries (LLM was adding `HAVING COUNT(*) > 1` to "most common" queries)
- Added few-shot examples for arrest queries, "most" queries, and tree neighborhood queries
- Added explicit instruction that `group_by` must be a list
- Added rules for crime_part handling ("Part 1 vs Part 2" is within-dataset, not a join)

### 3.4 LLM Output Validation

**Path 1 (Intent JSON)**:
- Schema validation enforces allowed values for every field
- Invalid datasets, metrics, or columns are rejected with clear error messages
- Filters are normalized to `{op, value}` format

**Path 2 (LLM SQL)**:
- `sql_validator.py` performs multiple safety checks:
  - **Read-only enforcement**: Rejects INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE
  - **Table allowlist**: Only 16 known tables permitted
  - **Semicolon rejection**: Blocks multi-statement injection
  - **LIMIT enforcement**: Injects LIMIT 1000 if absent, caps existing LIMIT at 1000
  - **CTE alias extraction**: Correctly identifies CTE names so they don't trigger the table allowlist check
  - **Dangerous pattern blocking**: Rejects GRANT, REVOKE, EXEC, and system catalog access

### 3.5 Heuristic Fallback

When no API key is available, `_heuristic_parse()` handles queries using:
- Dataset detection via keyword matching (e.g., "crime", "arrest", "offense" -> crime dataset)
- Metric detection ("average", "total", "unique")
- Group-by extraction ("by neighborhood", "by year")
- Filter extraction ("since 2020", "in 2024")
- HAVING detection ("more than", "over", "at least")
- Join detection ("with violations", "compare crime and")

This enables the app to function entirely without an LLM for common query patterns.

---

## 4. Validation Framework

### 4.1 Ground-Truth Validation (`validation.py`)

Every Path 1 query result is compared against an independent pandas calculation:

1. **Count validation**: The SQL `COUNT(*)` result is compared against `len(df)` (with matching filters applied via pandas).

2. **Grouped count validation**: For grouped queries, the SQL group counts are compared against `df.groupby(col).size()`. Discrepancies >5% are flagged as errors.

3. **Join validation**: Join results are compared against `pd.merge()` operations with the same join keys.

4. **Sanity checks** (`sanity_check_result()`):
   - Flags results with suspiciously high counts (>1M)
   - Warns about null groups in results
   - Detects outlier values in aggregation results
   - Checks for empty results that should have data

### 4.2 Validation Output

Each query response includes a validation object:
```json
{
  "passed": true,
  "warnings": ["17% of records have 'Unknown' neighborhood"],
  "errors": [],
  "ground_truth": {"total_count": 138245, "sql_count": 138245}
}
```

### 4.3 Path 2 Limitations

LLM-generated SQL (Path 2) does not receive ground-truth validation because the query structures (window functions, CTEs, subqueries) are too varied to systematically replicate in pandas. Instead, Path 2 results include a warning: "LLM-generated SQL - no ground-truth validation."

---

## 5. Bias Detection Framework

### 5.1 Bias Types

`bias_detection.py` checks for five categories of potential bias in query results:

| Bias Type | What It Detects | Example Warning |
|-----------|----------------|-----------------|
| **Framing** | Leading or loaded language in the question | "The question uses 'dangerous' which may frame interpretation" |
| **Normalization** | Raw counts that need per-capita or proportional context | "Raw counts may not reflect rates - consider normalizing by population" |
| **Selection** | Known limitations in how data was collected | "Crime data reflects reported incidents, not all crimes" |
| **Context** | Missing analytical context for proper interpretation | "Violation counts vary by inspector activity, not just actual conditions" |
| **Uncertainty** | Sources of uncertainty in the results | "2025 crime data is partial (33 records only)" |

### 5.2 Implementation

Each bias check is a pure function that receives the question, result DataFrame, intent, and metadata. Checks use keyword matching and dataset-specific rules rather than LLM inference, making them deterministic and fast.

The `BiasResult` object aggregates all warnings and is serialized to a list of `{type, message}` dicts in the API response.

### 5.3 Display

Bias warnings appear in the **Sources** tab of the web UI alongside data citations and caveats. This placement ensures transparency without cluttering the primary results view.

---

## 6. Limitations and Caveats

### 6.1 Data Limitations

- **Static snapshots**: All datasets are point-in-time CSV downloads, not live feeds. Data currency depends on when files were last updated.
- **Administrative records**: Datasets like violations and crime reflect enforcement/reporting activity, not ground truth. Under-reporting varies by type and neighborhood.
- **Partial years**: 2025 crime data contains only 33 records (January 1-5). The Syracuse Open Data Portal has not been updating regularly.
- **Unknown neighborhoods**: 17% of crime records (5,449) have "Unknown" neighborhood due to geocoding failures. These are pushed to the bottom of grouped results.
- **Privacy generalization**: Crime addresses are generalized to block level by the source data.
- **Assessment vs market value**: Property assessments may diverge significantly from actual market values.

### 6.2 Analytical Limitations

- **No causal inference**: The system reports correlations and distributions, not causes. "Neighborhoods with more violations" does not mean those neighborhoods have worse housing conditions - it may reflect more active enforcement.
- **No population normalization**: Raw counts are reported without per-capita adjustment. A neighborhood with more violations may simply have more housing units. The bias detection framework warns about this.
- **No temporal adjustment**: Crime trends are shown without seasonal adjustment or population change correction.
- **Subjective questions rejected**: Questions like "What is the safest neighborhood?" are explicitly declined with a message suggesting data-based alternatives.

### 6.3 System Limitations

- **Heuristic parser coverage**: The keyword-based parser handles ~80% of common queries but fails on unusual phrasing. The LLM parser is more flexible but requires an API key.
- **LLM SQL hallucination**: Path 2 SQL generation may produce semantically incorrect queries (correct syntax but wrong logic). The SQL validator catches safety issues but cannot verify semantic correctness.
- **No multi-turn conversation**: Each query is independent. The system cannot reference previous questions or build on prior results.
- **LIMIT 1000**: All queries are capped at 1000 rows to prevent excessive data transfer and display issues.
- **Single-language**: English only. No internationalization support.

### 6.4 Ethical Considerations

- Crime data can reinforce neighborhood stigmatization if presented without context. The bias detection framework mitigates this by warning about normalization needs.
- Lead testing data is census-tract level and intended for research use only. Individual-level conclusions should not be drawn.
- The system includes explicit data citations and caveats for every dataset to promote responsible interpretation.
