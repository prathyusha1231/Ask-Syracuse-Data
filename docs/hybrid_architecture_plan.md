# Plan: Hybrid Query Architecture — Expand Scope Beyond COUNT

## Context

The system currently only supports `COUNT + GROUP BY + equality filters`. Users can't ask temporal questions ("violations by year"), aggregate questions ("average days to comply"), distinct counts ("how many unique properties"), threshold queries ("neighborhoods with >100 violations"), or truly complex queries (rankings, percentiles, window functions). We're expanding scope with a **hybrid approach**: extend the deterministic intent pipeline for common patterns, and add an LLM-generated SQL path with guardrails for complex queries.

## Architecture Overview

```
User Question
    ↓
Intent Parser (detects complexity)
    ├─ Simple/Medium → Expanded Intent JSON → Schema Validator → SQL Builder (deterministic)
    │   Supports: count, count_distinct, avg/min/max on computed cols,
    │             temporal group-by, richer filters, HAVING, multi group-by
    │
    └─ Complex → LLM SQL Generation → SQL Validator (guardrails) → DuckDB
        Supports: window functions, CASE WHEN, subqueries, anything SELECT
```

Both paths output DataFrame + metadata → same response format to frontend.

## Files to Modify/Create

| File | Change |
|------|--------|
| `pipeline/schema.py` | Expand DATASETS config (metrics, temporal groups, computed columns, richer filters), update validators |
| `pipeline/sql_builder.py` | Handle new metrics, temporal GROUP BY, richer WHERE, HAVING, computed columns, multi group-by |
| `llm/intent_parser.py` | Add routing (`_needs_advanced_sql()`), expand heuristic for temporal/distinct/having |
| `llm/prompt_templates.py` | Expand intent prompt for new fields, add new SQL generation prompt |
| `llm/openai_client.py` | Add `make_openai_sql_llm()` for SQL generation path |
| `pipeline/main.py` | Add `_run_advanced_query()`, update router for 3 paths |
| `pipeline/sql_validator.py` | **NEW** — SQL guardrails (allowlist, read-only, LIMIT enforcement) |
| `pipeline/validation.py` | Generalize beyond count-only validation |
| `app.py` | Handle new metrics, line charts for temporal, advanced SQL results |
| `CLAUDE.md` | Update throughout with new capabilities |

## Progress Tracker

### Phase 1: Expand Deterministic Intent Pipeline
- [x] **Step 1**: Expand `pipeline/schema.py` — DONE
- [x] **Step 2**: Expand `pipeline/sql_builder.py` — DONE (composable helpers, temporal GROUP BY, HAVING, multi GROUP BY)
- [x] **Step 3**: Update `llm/prompt_templates.py` — DONE (expanded intent prompt + NL_TO_SQL_PROMPT)
- [x] **Step 4**: Update `llm/intent_parser.py` — DONE (temporal, distinct, HAVING, year filter detection + advanced SQL routing)
- [x] **Step 5**: Update `pipeline/validation.py` — DONE (count_distinct, operator filters, list group_by)
- [x] **Step 6**: Update `pipeline/main.py` — DONE (3-path routing, expanded metadata)
- [x] **Step 7**: Update `app.py` — DONE (line charts, new column labels, advanced SQL support)

### Phase 2: LLM SQL Generation Path
- [x] **Step 8**: Create `pipeline/sql_validator.py` — DONE (read-only, table allowlist, CTE support, LIMIT enforcement)
- [x] **Step 9**: Add `make_openai_sql_llm()` to `llm/openai_client.py` — DONE
- [x] **Step 10**: Add `_run_advanced_query()` in `pipeline/main.py` — DONE
- [x] **Step 11**: Update `app.py` for advanced SQL results — DONE (merged with Step 7)

### Verification
- [x] **Step 12**: Run tests and verify + update CLAUDE.md — DONE (42/42 tests pass, CLAUDE.md updated)

### Additional fixes applied during implementation:
- Fixed `bias_detection.py` to handle `group_by` as list (unhashable type error)
- Fixed `_is_join_query()` to exclude distinct/HAVING patterns from join routing
- Fixed CTE alias extraction in `sql_validator.py` for multi-CTE queries

## Phase 1: Expand Deterministic Intent Pipeline

### Step 1: Expand `pipeline/schema.py`

**1a. Add `allowed_metrics` to each dataset:**
```python
"violations": {
    ...
    "allowed_metrics": ["count", "count_distinct", "avg", "min", "max"],
}
"crime_2022": {
    ...
    "allowed_metrics": ["count", "count_distinct"],
}
```
Only violations and vacant_properties have date columns for avg/min/max. Crime and rental get count + count_distinct only.

**1b. Add `computed_columns` for date-diff metrics:**
```python
"violations": {
    "computed_columns": {
        "days_to_comply": {
            "expr": "date_diff('day', violation_date, comply_by_date)",
            "type": "numeric",
        },
        "days_open": {
            "expr": "date_diff('day', open_date, status_date)",
            "type": "numeric",
        },
    },
},
"vacant_properties": {
    "computed_columns": {
        "cert_duration_days": {
            "expr": "date_diff('day', completion_date, valid_until)",
            "type": "numeric",
        },
    },
},
```

**1c. Add temporal group-by support:**
```python
"violations": {
    "temporal_group_map": {
        "year": ("violation_date", "year"),
        "month": ("violation_date", "month"),
        "quarter": ("violation_date", "quarter"),
    },
    "allowed_group_by": ["neighborhood", "complaint_zip", "status_type_name", "violation",
                         "year", "month", "quarter"],
},
```

**1d. Expand filter format** — accept both old and new:
- Old: `{"year": 2020}` (equality, backward compat)
- New: `{"year": {"op": ">=", "value": 2020}}`
- Ops: `=`, `>=`, `<=`, `between`, `in`, `like`

**1e. Update `validate_intent()`:**
- Accept metrics from `allowed_metrics` (not hard-coded "count")
- Accept `group_by` as list or string (normalize to list)
- Accept `metric_column` for avg/min/max on computed columns
- Accept `distinct_column` for count_distinct
- Accept `having` clause: `{"op": ">", "value": 100}`
- Normalize old filter format to new format internally
- Keep backward compat: old intents pass through unchanged

**1f. Update `validate_join_intent()`:**
- Same metric expansion (accept more than just "count")

### Step 2: Expand `pipeline/sql_builder.py`

**2a. Refactor `build_select_sql()` into composable helpers:**
- `_build_metric_expr(intent, config)` → `"COUNT(*) AS count"` or `"AVG(date_diff(...)) AS avg_days_to_comply"` etc.
- `_build_group_exprs(group_by_list, config)` → handles temporal groups via `date_part()`
- `_build_where_clause(filters, config)` → handles `=`, `>=`, `<=`, `between`, `in`, `like`
- `_build_having_clause(having)` → `"HAVING COUNT(*) > 100"`
- Assemble in `build_select_sql()` using these helpers

**2b. Key SQL patterns generated:**

Temporal grouping:
```sql
SELECT date_part('year', violation_date) AS year, COUNT(*) AS count
FROM violations GROUP BY date_part('year', violation_date) ORDER BY year;
```

COUNT DISTINCT:
```sql
SELECT neighborhood, COUNT(DISTINCT sbl) AS count_distinct
FROM violations GROUP BY neighborhood ORDER BY count_distinct DESC;
```

Computed column AVG:
```sql
SELECT neighborhood, AVG(date_diff('day', violation_date, comply_by_date)) AS avg_days_to_comply
FROM violations WHERE violation_date IS NOT NULL AND comply_by_date IS NOT NULL
GROUP BY neighborhood ORDER BY avg_days_to_comply DESC;
```

HAVING:
```sql
SELECT neighborhood, COUNT(*) AS count FROM violations
GROUP BY neighborhood HAVING COUNT(*) > 100 ORDER BY count DESC;
```

Multiple group-by:
```sql
SELECT neighborhood, date_part('year', violation_date) AS year, COUNT(*) AS count
FROM violations GROUP BY neighborhood, date_part('year', violation_date)
ORDER BY count DESC;
```

BETWEEN filter:
```sql
SELECT neighborhood, COUNT(*) AS count FROM violations
WHERE date_part('year', violation_date) BETWEEN 2020 AND 2023
GROUP BY neighborhood ORDER BY count DESC;
```

### Step 3: Update `llm/prompt_templates.py`

**3a. Expand `NL_TO_INTENT_PROMPT`** — teach the LLM the new JSON fields:
- Show the expanded intent schema with `metric`, `metric_column`, `distinct_column`, `group_by` as array, `having`, richer filters with ops
- List computed columns per dataset
- List temporal group options per dataset
- Add 4-5 examples of new query types

**3b. Expand `NL_TO_JOIN_INTENT_PROMPT`** similarly (add metrics beyond count for joins).

**3c. Add `NL_TO_SQL_PROMPT`** for Path 2 (Phase 2):
- Full schema with all table names, column names, types
- DuckDB SQL syntax rules
- Safety rules (SELECT only, must have LIMIT)

### Step 4: Update `llm/intent_parser.py`

**4a. Expand heuristics** to detect new patterns:
- Temporal: "by year", "per year", "monthly", "yearly", "over time", "trend"
- Distinct: "unique", "distinct", "different"
- Having: "more than N", "over N", "at least N", "fewer than N"
- Year ranges: "since 2020", "between 2020 and 2023", "after 2020"
- Computed: "average days", "avg time", "how long", "fastest", "slowest"

**4b. Add `_needs_advanced_sql()` detector** for Path 2:
- Keywords: "rank", "percentile", "top N percent", "running total", "cumulative", "compared to average", "above/below average", "rate per", "ratio", "percent change", "rolling", "moving average"

**4c. Update `parse_intent()` to return routing signal:**
- If `_needs_advanced_sql()` → return `{"query_path": "advanced_sql", "question": question}`
- Otherwise → existing intent parsing

### Step 5: Update `pipeline/validation.py`

**5a. Generalize `validate_count_result()`** to `validate_metric_result()`:
- COUNT: existing logic (pandas groupby().size())
- COUNT DISTINCT: pandas nunique()
- AVG/MIN/MAX: pandas agg() on computed column equivalent
- Keep `validate_count_result` as alias for backward compat

**5b. Update `_apply_filters()`** to handle new filter format with operators.

### Step 6: Update `pipeline/main.py`

**6a.** In `_run_single_query()`, pass through new intent fields to metadata.
**6b.** Call `validate_metric_result()` instead of `validate_count_result()` (alias means no breakage).
**6c.** Add metadata fields: `metric_column`, `distinct_column`, `having`.

### Step 7: Update `app.py`

**7a.** Add column labels for new output columns (year, month, quarter, count_distinct, avg_*, min_*, max_*).
**7b.** Add line chart type for temporal data (when group_by includes year/month/quarter).
**7c.** Update `generate_description()` for new metrics.

## Phase 2: LLM SQL Generation Path

### Step 8: Create `pipeline/sql_validator.py` (NEW)

Guardrails for LLM-generated SQL:
- Must start with SELECT or WITH (CTEs allowed)
- Reject write ops: INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT
- Reject system access: information_schema, pg_*, read_csv, httpfs
- Table allowlist: only {violations, rental_registry, vacant_properties, crime_2022}
- LIMIT enforcement: inject LIMIT 1000 if missing, cap at 1000
- Extract table names from FROM/JOIN clauses and validate

### Step 9: Add SQL generation to `llm/openai_client.py`

Add `make_openai_sql_llm()` — same as intent LLM but:
- No JSON response format constraint
- Higher max_tokens (500 vs 300)
- System message: "You are a SQL query generator. Return only valid DuckDB SQL."

### Step 10: Add `_run_advanced_query()` in `pipeline/main.py`

Flow:
1. Call SQL LLM with `NL_TO_SQL_PROMPT.format(question=question)`
2. Strip markdown fences from response
3. Run through `sql_validator.validate_sql()`
4. Load all 4 tables into DuckDB
5. Execute validated SQL
6. Return with metadata `{"query_type": "advanced_sql"}`
7. Sanity checks only (no ground-truth validation for arbitrary SQL)
8. Add limitation text: "This query was generated by an LLM. Results should be verified."

Update `run_query()` router:
```python
if raw_intent.get("query_path") == "advanced_sql":
    response = _run_advanced_query(question)
elif raw_intent.get("query_type") == "join":
    response = _run_join_query(raw_intent)
else:
    response = _run_single_query(raw_intent)
```

### Step 11: Update `app.py` for advanced SQL results

- Handle `query_type == "advanced_sql"` in description generator
- Auto-detect chart type from result columns (text col → x-axis, numeric col → y-axis)
- Show "LLM-generated query" indicator in metadata

## Verification

1. Run `python -m tests.eval_benchmarks` — all 12 existing tests must pass
2. Run `python -m tests.test_app_comprehensive` — all 30 existing tests must pass (29 + test 26 flaky)
3. Test new Path 1 queries manually:
   - "violations by year" → line chart with yearly counts
   - "how many unique properties have violations" → count_distinct
   - "neighborhoods with more than 100 violations" → filtered by HAVING
   - "average days to comply by neighborhood" → avg computed column
   - "violations by neighborhood and year" → multi group-by
   - "violations since 2020" → year >= filter
   - "violations between 2020 and 2023" → BETWEEN filter
4. Test new Path 2 queries manually:
   - "rank neighborhoods by violations" → LLM SQL with window function
   - Verify guardrails reject "DELETE FROM violations"
5. Update CLAUDE.md after each phase
