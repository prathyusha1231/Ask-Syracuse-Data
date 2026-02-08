# Improvements Tracker

## Critical (Security & Availability)

- [x] **1. Fix XSS vulnerabilities** — innerHTML with unsanitized LLM/backend content in insights, bias warnings, citations
- [x] **2. Add rate limiting** — `/api/query` is unprotected, OpenAI cost exposure
- [x] **3. Add SQL execution timeout + semicolon rejection** — prevent hanging queries and multi-statement injection
- [x] **4. Add input validation on QueryRequest** — no length limits on user questions

## High Priority (Reliability & Correctness)

- [x] **5. Fix limitations string for Path 2** — falsely claims "LLM used only for intent parsing"
- [x] **6. Add try/except to feedback DB operations** — crashes on locked/corrupt SQLite
- [x] **7. Replace generic Exception catching in main.py** — swallows KeyboardInterrupt, hides debug info
- [x] **8. Add query logging/audit trail** — no record of executed SQL

## Medium Priority (Testing)

- [x] **9. Migrate tests to pytest** — custom harnesses, no CI/CD support
- [x] **10. Add metric/HAVING/filter tests** — zero tests for avg/min/max, HAVING, >=, BETWEEN
- [x] **11. Add error handling / negative tests** — no tests for bad input, expected failures
- [x] **12. Add ground-truth validation in tests** — tests check structure not correctness

## Lower Priority (Defense in Depth)

- [x] **13. Add SRI hashes to CDN resources** — Plotly integrity hash added
- [x] **14. Add Content-Security-Policy header** — CSP + X-Frame-Options + X-Content-Type-Options
- [x] **15. Add retry logic to OpenAI client** — exponential backoff on rate limit, timeout, server errors
- [x] **16. Add DataFrame cache invalidation** — 1-hour TTL on cached DataFrames
- [x] **17. Add frontend loading timeout** — 60s AbortController timeout with user-friendly message
- [x] **18. Improve accessibility** — ARIA labels on search input, submit button, feedback buttons

## Status

| # | Item | Status | Date |
|---|------|--------|------|
| 1 | XSS fixes | Done | 2026-02-08 |
| 2 | Rate limiting | Done | 2026-02-08 |
| 3 | SQL timeout + semicolons | Done | 2026-02-08 |
| 4 | Input validation | Done | 2026-02-08 |
| 5 | Limitations string | Done | 2026-02-08 |
| 6 | Feedback DB error handling | Done | 2026-02-08 |
| 7 | Exception handling + logging | Done | 2026-02-08 |
| 8 | Query logging | Done | 2026-02-08 |
| 9 | Pytest migration | Done | 2026-02-08 |
| 10 | Metric/HAVING tests | Done | 2026-02-08 |
| 11 | Negative tests | Done | 2026-02-08 |
| 12 | Ground-truth tests | Done | 2026-02-08 |
| 13 | SRI hashes | Done | 2026-02-08 |
| 14 | CSP header | Done | 2026-02-08 |
| 15 | OpenAI retry logic | Done | 2026-02-08 |
| 16 | Cache invalidation | Done | 2026-02-08 |
| 17 | Frontend timeout | Done | 2026-02-08 |
| 18 | Accessibility | Done | 2026-02-08 |

## Changes Log

### 2026-02-08: Items 1-8 (Critical + High Priority)

**Files changed:**
- `templates/index.html` — Added `escapeHtml()` and `sanitizeUrl()` helpers; escaped all dynamic content in insights, validation, bias warnings, and citations
- `app.py` — Added slowapi rate limiting (20/min query, 30/min feedback), input validation (max 500 chars), feedback DB try/except/finally, timezone-aware timestamps
- `pipeline/sql_validator.py` — Added semicolon rejection to block multi-statement SQL injection
- `pipeline/main.py` — Added 30s SQL timeout via ThreadPoolExecutor, `logger.exception()` on all catch blocks, query audit logging (question, path, status, rows, time, SQL)
- `requirements.txt` — Added `slowapi>=0.1.9`

### 2026-02-08: Items 9-12 (Testing)

**Files created:**
- `tests/test_pipeline.py` — 73 pytest tests across 10 test classes
- `tests/conftest.py` — Auto-disables LLM for tests (forces heuristic mode); set `FORCE_LLM_TESTS=1` to use real API

**Test coverage:**
- `TestIntentParsing` (15) — heuristic parsing for all datasets, joins, filters, metrics
- `TestMetricQueries` (4) — count, parking fines, cityline categories
- `TestTemporalQueries` (3) — violations/crime by year
- `TestHavingQueries` (3) — threshold filtering ("more than N", "over N")
- `TestFilterQueries` (3) — year equals, year since, ground-truth validation
- `TestJoinQueries` (3) — ZIP and SBL cross-dataset joins
- `TestErrorHandling` (6) — empty, nonsense, long, XSS, SQL injection, data survival
- `TestSQLValidator` (13) — DROP/INSERT/DELETE, semicolons, unknown tables, LIMIT enforcement
- `TestSchemaValidation` (5) — valid/invalid datasets, metrics, normalization
- `TestGroundTruth` (4) — total counts and grouped counts vs pandas
- `TestAllDatasets` (14) — parametrized smoke test for all 14 parseable datasets

### 2026-02-08: Items 13-18 (Defense in Depth)

**Files changed:**
- `templates/index.html` — Plotly SRI hash, 60s fetch timeout with AbortController, ARIA labels on input/buttons
- `app.py` — SecurityHeadersMiddleware (CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy)
- `llm/openai_client.py` — `_call_with_retry()` with exponential backoff (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError), 30s client timeout
- `pipeline/main.py` — `_get_cached_df()` with 1-hour TTL replacing raw dict cache
- `requirements.txt` — Added `pytest>=7.0.0`

**Verified:** 73/73 pytest tests pass + 13/13 benchmark tests pass.
