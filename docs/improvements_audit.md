# Improvements Audit — Ask Syracuse Data

Full repository audit covering security, performance, code quality, testing, UX, data quality, and architecture. Findings are prioritized as **High**, **Medium**, or **Low**.

---

## 1. Security

### High Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| S1 | SQL column injection via unquoted identifiers | `pipeline/sql_builder.py:34` | `distinct_column` from intent inserted directly into SQL without quoting. If LLM returns a crafted column name, SQL injection possible. |
| S2 | f-string SQL construction with user-influenced values | `pipeline/sql_builder.py:67,104,107` | `col_expr`, `date_col`, and temporal map values constructed via f-strings. If config values contain SQL syntax, injection possible. |
| S3 | Semicolon check bypassable | `pipeline/sql_validator.py:157-162` | Splits on `;` but `CHAR(59)` or unicode escapes could bypass. |
| S4 | Comment injection not fully blocked | `pipeline/sql_validator.py:177` | `\b` word boundary in regex doesn't catch `DROP/**/TABLE` (comment-separated keywords). |

### Medium Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| S5 | `_quote()` uses simple quote-doubling | `pipeline/sql_builder.py:13` | Insufficient for DuckDB-specific escape sequences like `E'\\'`. |
| S6 | CTE alias extraction incomplete | `pipeline/sql_validator.py:188` | Malformed CTEs could escape detection — doesn't validate CTEs are properly closed. |
| S7 | No ZIP validation on extracted values | `pipeline/data_utils.py:352` | Regex `r'(\d{5})\s*$'` doesn't validate ZIP is one of the 12 valid Syracuse ZIP codes. |

---

## 2. Performance

### Medium Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| P1 | All datasets loaded for advanced SQL queries | `pipeline/main.py:391-392` | Loads all 16 datasets even if query uses only 1-2. Scales poorly as datasets grow. Could lazy-load based on table names extracted from SQL. |
| P2 | ZIP assignment creates full distance matrix | `pipeline/data_utils.py:181` | `np.sqrt()` in `_assign_zip_from_coords()` creates (N, 12, 2) array for every record. For ~32K crime rows, this is a 32K x 12 distance matrix. Could use `scipy.spatial.distance.cdist`. |
| P3 | `pd.concat(frames)` without `sort=False` | `pipeline/data_utils.py:281` | Sorts columns by default — unnecessary work during crime CSV concatenation. |

### Low Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| P4 | Neighborhood normalization chains 3 Pandas ops | `pipeline/data_utils.py:233-234` | `.str.lower()` then `.map()` then `.fillna()` — could combine with single `.apply()`. |
| P5 | Validation `value_counts()` then iteration | `pipeline/validation.py:298-302` | Could use groupby merge pattern for better performance on large datasets. |

---

## 3. Code Quality

### Error Handling

| # | Finding | File | Details |
|---|---------|------|---------|
| Q1 | Bare `except Exception` masks specific errors | `pipeline/main.py:126-128` | LLM initialization catch-all could mask ImportError, AttributeError. |
| Q2 | `choices[0]` assumed safe without guard | `llm/openai_client.py:79,107` | `completion.choices[0]` could raise IndexError if OpenAI returns empty response. |
| Q3 | Filter application silently swallows errors | `pipeline/validation.py:99-106,112` | `except Exception: pass` in filter application ignores invalid operators or type mismatches. |
| Q4 | JSONDecodeError catch doesn't strip markdown | `llm/intent_parser.py:746-748` | LLM output may contain markdown fences before/after JSON — no stripping of output. |
| Q5 | CSV reading has no timeout | `pipeline/data_utils.py:345` | `pd.read_csv(path, low_memory=False)` could hang indefinitely on corrupted files. |

### Type Hints

| # | Finding | File | Details |
|---|---------|------|---------|
| Q6 | Missing return type hints | Multiple | `validation.py:36-114` (`_apply_filters`), `bias_detection.py:52+` (all detect functions), `intent_parser.py:164` (`_detect_metric`). |
| Q7 | Missing parameter type hints | `llm/intent_parser.py:215-338` | `_heuristic_join_intent()` missing parameter types entirely. |

### Code Duplication

| # | Finding | File | Details |
|---|---------|------|---------|
| Q8 | ZIP and neighborhood join SQL nearly identical | `pipeline/sql_builder.py:245-284,345-384` | `_build_zip_join_sql()` and `_build_neighborhood_join_sql()` are ~99% identical. Could extract parameterized helper. |
| Q9 | Single-dataset intent builders repeated 3x | `llm/intent_parser.py:366-456` | Violations, vacant, rental intent builders have identical structure. Could extract `_build_single_dataset_intent()`. |
| Q10 | Loader functions follow identical pattern | `pipeline/data_utils.py:119-150` | `_load_csv()` + `_normalize_sbl()` + `_apply_null_handling()` repeated for each dataset. Could consolidate with dataset-specific config. |

---

## 4. Testing

### High Priority

| # | Finding | Details |
|---|---------|---------|
| T1 | No unit tests for `sql_validator.validate_sql()` | Dangerous pattern detection, LIMIT injection, and bypass attempts are untested. |
| T2 | No unit tests for `sql_builder.py` SQL generation | Only integration tests exist — individual SQL generation helpers untested. |
| T3 | Security bypass payloads untested | No tests for SQL injection payloads, comment injection, unicode escapes. |

### Medium Priority

| # | Finding | Details |
|---|---------|---------|
| T4 | No tests for `data_utils.py` edge cases | ZIP derivation, neighborhood normalization, null handling edge cases (empty DataFrames, all-null columns). |
| T5 | No tests for `validation.py` filter application | Group-by resolution, filter type coercion, operator combinations (BETWEEN, LIKE, IN). |
| T6 | No tests for `bias_detection.py` individual detectors | Each bias type (framing, normalization, selection, context, uncertainty) untested in isolation. |
| T7 | Exception paths untested | LLM init failure, SQL execution timeout, validation errors in `pipeline/main.py`. |
| T8 | HAVING clause edge cases untested | `sql_builder.py:159-165` — HAVING with computed columns, multiple conditions. |

### Low Priority

| # | Finding | Details |
|---|---------|---------|
| T9 | Cache TTL behavior untested | `pipeline/main.py:77-87` — no tests for cache expiry or invalidation. |
| T10 | Multi-language SQL comments in validator | `sql_validator.py:177` regex not tested with non-ASCII comments. |

---

## 5. UX & Accessibility

### Medium Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| U1 | No ARIA live region for results | `templates/index.html` | Screen readers won't announce when results load. Add `aria-live="polite"` to results container. |
| U2 | Accordion keyboard navigation incomplete | `templates/index.html` | Dataset category accordions and tech accordions lack `aria-expanded`, `aria-controls`, and `role="region"` attributes. |
| U3 | Table lacks `<caption>` element | `templates/index.html` | Data table has no caption — screen readers can't identify table purpose. |
| U4 | Color contrast on pill buttons | `templates/index.html` | `text-white/90` on `bg-white/15` may not meet WCAG AA contrast ratio (4.5:1) for small text. |

### Low Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| U5 | No loading state for individual grid cards | `templates/index.html` | When results load, all cards appear simultaneously. Staggered reveal would improve perceived performance. |
| U6 | No "no results" state | `templates/index.html` | If query returns 0 rows, the data table is empty with no message. |

---

## 6. Data Quality

### Medium Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| D1 | Crime+vacant join hardcodes 2024 filter | `pipeline/main.py:297-302` | Breaks if data is refreshed with newer years. Should use max year from data. |
| D2 | `pd.to_numeric(errors="coerce")` hides bad data | `pipeline/data_utils.py:410` | Silently converts invalid parking amounts to NaN with no logging or count of coerced values. |
| D3 | Crime lat/long rename fails silently | `pipeline/data_utils.py:264-266` | If only one of lat/long exists, rename silently fails — partial geolocation data goes undetected. |
| D4 | Computed column validation skipped | `pipeline/validation.py:175-178` | Avg/min/max on computed columns (days_to_comply, etc.) logged as warning but never validated against ground truth. |

### Low Priority

| # | Finding | File | Details |
|---|---------|------|---------|
| D5 | No data freshness tracking | Multiple | No metadata about when CSVs were downloaded or last updated. Users can't know data currency. |
| D6 | Lead testing year conversion is redundant | `pipeline/data_utils.py:456` | Year columns converted to string then back to numeric — unnecessary double conversion. |

---

## 7. Architecture

### Medium Priority

| # | Finding | Details |
|---|---------|---------|
| A1 | Schema config and loaders not validated against each other | `pipeline/schema.py` lists allowed joins and tables, but no runtime check that corresponding loader functions exist in `data_utils.py`. Adding a table to schema without a loader fails at query time. |
| A2 | Prompt templates tightly coupled to schema | `llm/prompt_templates.py` must manually match `pipeline/schema.py` — no auto-generation of prompts from schema config. Schema changes require updating 2+ files. |
| A3 | `group_by` normalization inconsistency | `pipeline/main.py:336` assigns group_by for join queries directly without list normalization, while `schema.py` normalizes it everywhere else. |
| A4 | No configuration file for thresholds | Bias detection thresholds (>10K records), validation tolerance (>5%), LIMIT cap (1000) are all hardcoded across multiple files. |

### Low Priority

| # | Finding | Details |
|---|---------|---------|
| A5 | No structured logging | Uses `print()` throughout. No log levels, no structured output, no log aggregation support. |
| A6 | No health check for data availability | `/api/health` endpoint doesn't verify that CSV files are loadable or that DuckDB can be initialized. |

---

## Summary

| Category | High | Medium | Low | Total |
|----------|------|--------|-----|-------|
| Security | 4 | 3 | 0 | 7 |
| Performance | 0 | 3 | 2 | 5 |
| Code Quality | 0 | 5 (errors) + 3 (duplication) | 2 (types) | 10 |
| Testing | 3 | 5 | 2 | 10 |
| UX | 0 | 4 | 2 | 6 |
| Data Quality | 0 | 4 | 2 | 6 |
| Architecture | 0 | 4 | 2 | 6 |
| **Total** | **7** | **31** | **12** | **50** |

### Recommended Priority Order

1. **S1-S4**: SQL injection vectors in sql_builder and sql_validator bypass (security)
2. **T1-T3**: Unit tests for SQL validation, generation, and security payloads (testing)
3. **Q1-Q2**: Empty response guards and specific exception types (error handling)
4. **P1**: Lazy-load datasets for advanced SQL path (performance)
5. **D1**: Remove hardcoded 2024 crime year filter (data quality)
6. **U1-U2**: ARIA attributes for accessibility compliance (UX)
7. **Q8-Q10**: Deduplicate join builders and intent builders (code quality)
8. **A4-A5**: Extract hardcoded thresholds and add structured logging (architecture)
