"""
CLI entry point for Ask Syracuse Data.
Pipeline: question -> LLM intent -> validate -> SQL -> DuckDB -> result.
Supports both single-dataset queries and cross-dataset joins.
Includes validation against ground-truth and bias detection.
"""
from __future__ import annotations
import re
import sys
import time
import logging
import threading
from collections import OrderedDict
import duckdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

SQL_TIMEOUT_SECONDS = 30
_RESULT_CACHE_MAX = 128
_RESULT_CACHE_TTL = 300  # 5 minutes

from . import schema
from .data_utils import (
    load_code_violations,
    load_rental_registry,
    load_vacant_properties,
    load_crime,
    load_unfit_properties,
    load_trash_pickup,
    load_historical_properties,
    load_assessment_roll,
    load_cityline_requests,
    load_snow_routes,
    load_bike_suitability,
    load_bike_infrastructure,
    load_parking_violations,
    load_permit_requests,
    load_tree_inventory,
    load_lead_testing,
    load_population,
)
from llm.intent_parser import parse_intent
from llm.openai_client import make_openai_intent_llm, make_openai_sql_llm, load_api_key
from llm.prompt_templates import NL_TO_SQL_PROMPT
from .security import assess_prompt_injection, SUSPICIOUS_LIMITATION_NOTE
from .sql_builder import build_select_sql, build_join_sql
from .sql_validator import validate_sql, SQLValidationError
from .validation import (
    validate_count_result,
    validate_join_result,
    sanity_check_result,
    combine_validations,
)
from .bias_detection import run_all_bias_checks


LOADERS = {
    "violations": load_code_violations,
    "vacant_properties": load_vacant_properties,
    "crime": load_crime,
    "rental_registry": load_rental_registry,
    "unfit_properties": load_unfit_properties,
    "trash_pickup": load_trash_pickup,
    "historical_properties": load_historical_properties,
    "assessment_roll": load_assessment_roll,
    "cityline_requests": load_cityline_requests,
    "snow_routes": load_snow_routes,
    "bike_suitability": load_bike_suitability,
    "bike_infrastructure": load_bike_infrastructure,
    "parking_violations": load_parking_violations,
    "permit_requests": load_permit_requests,
    "tree_inventory": load_tree_inventory,
    "lead_testing": load_lead_testing,
}

# Cache for loaded dataframes (for validation)
# Entries: {dataset_name: (dataframe, load_timestamp)}
_df_cache = {}
_CACHE_TTL_SECONDS = 3600  # Reload data after 1 hour

# Result cache for repeated queries
_result_cache: OrderedDict = OrderedDict()
_result_cache_lock = threading.Lock()


def _normalize_question(q: str) -> str:
    """Normalize a question string for cache key purposes."""
    return re.sub(r"\s+", " ", q.strip().lower())


def _get_cached_result(question: str) -> dict | None:
    """Return cached result if fresh, else None."""
    key = _normalize_question(question)
    with _result_cache_lock:
        entry = _result_cache.get(key)
        if entry is None:
            return None
        result, ts = entry
        if time.time() - ts > _RESULT_CACHE_TTL:
            _result_cache.pop(key, None)
            return None
        # Move to end (most recently used)
        _result_cache.move_to_end(key)
        return result


def _store_cached_result(question: str, result: dict):
    """Store a query result in cache (without raw_df to save memory)."""
    key = _normalize_question(question)
    # Strip raw_df (large DataFrame) before caching
    cached = {k: v for k, v in result.items() if k != "raw_df"}
    # Copy the result DataFrame to avoid mutation issues
    if isinstance(cached.get("result"), pd.DataFrame):
        cached["result"] = cached["result"].copy()
    with _result_cache_lock:
        _result_cache[key] = (cached, time.time())
        # Evict oldest if over limit
        while len(_result_cache) > _RESULT_CACHE_MAX:
            _result_cache.popitem(last=False)


def _get_cached_df(name: str, loader):
    """Get a DataFrame from cache, reloading if TTL expired."""
    now = time.time()
    if name in _df_cache:
        df, loaded_at = _df_cache[name]
        if now - loaded_at < _CACHE_TTL_SECONDS:
            return df
        logger.info("Cache expired for %s, reloading", name)
    df = loader()
    _df_cache[name] = (df, now)
    return df


def _make_error_response(error: str, **kwargs) -> dict:
    """Create a standard error response dict."""
    return {
        "result": kwargs.get("result"),
        "intent": kwargs.get("intent"),
        "metadata": kwargs.get("metadata", {}),
        "sql": kwargs.get("sql"),
        "error": error,
        "limitations": kwargs.get("limitations", ""),
        "validation": kwargs.get("validation"),
        "bias_warnings": kwargs.get("bias_warnings", []),
    }


def _execute_sql_with_timeout(conn, sql: str, timeout: int = SQL_TIMEOUT_SECONDS) -> pd.DataFrame:
    """Execute SQL on a DuckDB connection with a timeout guard."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: conn.execute(sql).df())
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            conn.close()
            raise TimeoutError(
                f"SQL query exceeded {timeout}s timeout. Try a simpler question."
            )


def _wants_per_capita(question: str) -> bool:
    """Check if question asks for per-capita / rate normalization."""
    q = question.lower()
    return any(phrase in q for phrase in ["per capita", "per person", "per resident", "rate per"])


def _add_per_capita_rate(result_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Add rate_per_1000 column by joining with population data."""
    group_by_str = metadata.get("group_by")
    if not group_by_str:
        return result_df

    # Determine which column to join on
    group_list = [g.strip() for g in str(group_by_str).split(",")]
    join_col = None
    pop_col = None
    for g in group_list:
        if g in ("zip", "complaint_zip"):
            join_col = g
            pop_col = "zip"
            break
        if g in ("neighborhood", "area"):
            join_col = g
            pop_col = "neighborhood"
            break

    if not join_col:
        return result_df

    try:
        pop_df = load_population()
    except FileNotFoundError:
        logger.warning("Population data not found, skipping per-capita calculation")
        return result_df

    # Find the count/value column
    value_col = None
    for col in ["count", "count_distinct", "avg", "sum", "min", "max"]:
        if col in result_df.columns:
            value_col = col
            break
    if not value_col:
        return result_df

    # Aggregate population by the join column (in case multiple ZIPs map to same neighborhood)
    if pop_col == "neighborhood":
        pop_agg = pop_df.groupby("neighborhood", as_index=False)["population"].sum()
    else:
        pop_agg = pop_df[["zip", "population"]].copy()

    # Merge
    df = result_df.copy()
    if join_col == "complaint_zip":
        df["_join_key"] = pd.to_numeric(df[join_col], errors="coerce").astype(pd.Int64Dtype())
        merged = df.merge(pop_agg, left_on="_join_key", right_on=pop_col, how="left")
        merged = merged.drop(columns=["_join_key", pop_col], errors="ignore")
    elif pop_col == "neighborhood":
        merged = df.merge(pop_agg, left_on=join_col, right_on="neighborhood", how="left")
        if join_col != "neighborhood":
            merged = merged.drop(columns=["neighborhood"], errors="ignore")
    else:
        df["_join_key"] = pd.to_numeric(df[join_col], errors="coerce").astype(pd.Int64Dtype())
        merged = df.merge(pop_agg, left_on="_join_key", right_on="zip", how="left")
        merged = merged.drop(columns=["_join_key"], errors="ignore")
        if join_col != "zip":
            merged = merged.drop(columns=["zip"], errors="ignore")

    # Calculate rate per 1,000
    if "population" in merged.columns:
        valid_pop = merged["population"].notna() & (merged["population"] > 0)
        merged.loc[valid_pop, "rate_per_1000"] = (
            merged.loc[valid_pop, value_col] / merged.loc[valid_pop, "population"] * 1000
        ).round(2)
        merged = merged.drop(columns=["population"])

    return merged


def run_query(question: str) -> dict:
    """Run the full pipeline and return a structured response."""
    t0 = time.perf_counter()
    logger.info("Query received: %s", question[:200])

    security = assess_prompt_injection(question)
    if security.status == "blocked":
        logger.warning("Blocked prompt-injection attempt: %s", security.reason)
        return _make_error_response(
            security.user_message or "Request blocked by security policy.",
            metadata={"security": security.to_metadata()},
            limitations="Request blocked by prompt-injection guard.",
        )

    # Check result cache
    cached = _get_cached_result(question)
    if cached is not None:
        logger.info("Cache hit for: %s", question[:100])
        # Avoid mutating the cached entry in-place.
        cached_copy = dict(cached)
        meta = dict(cached_copy.get("metadata") or {})
        meta["cache_hit"] = True
        meta["security"] = security.to_metadata()
        cached_copy["metadata"] = meta
        return cached_copy

    llm_fn = None
    api_key = load_api_key()
    if api_key and security.status != "suspicious":
        try:
            llm_fn = make_openai_intent_llm()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize LLM client")
            return _make_error_response(f"Unable to initialize LLM client: {exc}")

    raw_intent = None
    try:
        raw_intent = parse_intent(question, llm=llm_fn)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse intent for: %s", question)
        note = " (GPT disabled; set OPENAI_API_KEY to enable)" if llm_fn is None else ""
        return _make_error_response(f"Unable to parse intent: {exc}{note}", intent=raw_intent)

    # Check if the parser is asking for location clarification
    if raw_intent.get("needs_clarification"):
        return {
            "result": None,
            "intent": raw_intent,
            "metadata": {},
            "sql": None,
            "error": None,
            "clarification": raw_intent,
        }

    # Route to the appropriate query handler
    if raw_intent.get("query_path") == "advanced_sql":
        response = _run_advanced_query(raw_intent["question"])
    elif raw_intent.get("query_type") == "join":
        response = _run_join_query(raw_intent)
    else:
        response = _run_single_query(raw_intent)

    response.setdefault("metadata", {})
    meta = dict(response.get("metadata") or {})
    meta["cache_hit"] = False
    meta["security"] = security.to_metadata()
    meta["intent_parser"] = "llm" if llm_fn else "heuristic"
    if meta.get("query_type") == "advanced_sql":
        meta["sql_generator"] = "llm"
    else:
        meta["sql_generator"] = "deterministic"
    if raw_intent.get("query_path") == "advanced_sql" and raw_intent.get("routing_reasons"):
        meta["routing_reasons"] = raw_intent.get("routing_reasons")
    response["metadata"] = meta

    # Preserve advanced-sql routing reasons in the response intent for transparency.
    if raw_intent.get("query_path") == "advanced_sql" and isinstance(response.get("intent"), dict):
        response["intent"].setdefault("routing_reasons", raw_intent.get("routing_reasons") or [])

    if security.status != "safe":
        if response.get("limitations"):
            response["limitations"] += " " + SUSPICIOUS_LIMITATION_NOTE
        else:
            response["limitations"] = SUSPICIOUS_LIMITATION_NOTE

    # Add per-capita rates if requested
    if (response.get("result") is not None
            and response.get("error") is None
            and _wants_per_capita(question)):
        response["result"] = _add_per_capita_rate(
            response["result"], response.get("metadata", {})
        )
        if "rate_per_1000" in response["result"].columns:
            response["metadata"]["has_per_capita"] = True

    # Run bias detection on successful queries
    if response.get("result") is not None and response.get("error") is None:
        bias_result = run_all_bias_checks(
            question,
            response["result"],
            response.get("intent", {}),
            response.get("metadata", {}),
        )
        response["bias_warnings"] = bias_result.to_list()

    elapsed = time.perf_counter() - t0
    path = response.get("metadata", {}).get("query_type", "unknown")
    rows = response.get("metadata", {}).get("row_count", 0)
    status = "error" if response.get("error") else "ok"
    logger.info(
        "Query complete: path=%s status=%s rows=%s time=%.2fs sql=%s",
        path, status, rows, elapsed,
        (response.get("sql") or "")[:120],
    )

    # Cache successful results
    if response.get("error") is None:
        _store_cached_result(question, response)

    return response


def _group_by_for_metadata(group_by) -> str | None:
    """Convert group_by (list or None) to display-friendly string for metadata."""
    if group_by is None:
        return None
    if isinstance(group_by, list):
        return group_by[0] if len(group_by) == 1 else ", ".join(group_by)
    return group_by


def _run_single_query(raw_intent: dict) -> dict:
    """Execute a single-dataset query with validation."""
    try:
        intent = schema.validate_intent(raw_intent)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Schema validation failed")
        return _make_error_response(f"Validation failed: {exc}", intent=raw_intent)

    dataset = intent["dataset"]
    cfg = schema.get_dataset_config(dataset)
    loader = LOADERS.get(dataset)
    if not loader:
        return _make_error_response(
            f"No loader found for dataset '{dataset}'.", intent=intent
        )

    try:
        # Load data (cache with TTL for validation)
        df = _get_cached_df(dataset, loader)

        conn = duckdb.connect(database=":memory:")
        conn.register(cfg["table"], df)
        sql = build_select_sql(intent, cfg)
        result_df = _execute_sql_with_timeout(conn, sql)
    except Exception as exc:  # noqa: BLE001
        logger.exception("SQL execution failed for single query")
        return _make_error_response(
            f"Failed to execute SQL: {exc}",
            intent=intent,
            metadata={"intent": intent},
            sql=sql if "sql" in locals() else None,
        )

    limitations = (
        "Static CSV snapshots only; DuckDB queries are deterministic. "
        "If LLM was used, it was only for intent parsing; it never accesses data directly."
    )

    metadata = {
        "query_type": "single",
        "dataset": dataset,
        "filters": intent.get("filters") or {},
        "group_by": _group_by_for_metadata(intent.get("group_by")),
        "metric": intent.get("metric", "count"),
        "metric_column": intent.get("metric_column"),
        "distinct_column": intent.get("distinct_column"),
        "having": intent.get("having"),
        "limit": intent.get("limit"),
        "row_count": len(result_df),
    }

    # Run validation against ground-truth
    count_validation = validate_count_result(result_df, df, intent, metadata)
    sanity_validation = sanity_check_result(result_df, intent, metadata)
    validation = combine_validations(count_validation, sanity_validation)

    return {
        "result": result_df,
        "raw_df": df,
        "intent": intent,
        "metadata": metadata,
        "limitations": limitations,
        "sql": sql,
        "error": None,
        "validation": validation.to_dict(),
        "bias_warnings": [],
    }


def _run_join_query(raw_intent: dict) -> dict:
    """Execute a cross-dataset join query with validation."""
    try:
        intent = schema.validate_join_intent(raw_intent)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Join validation failed")
        return _make_error_response(f"Join validation failed: {exc}", intent=raw_intent)

    primary = intent["primary_dataset"]
    secondary = intent["secondary_dataset"]
    join_type = intent["join_type"]

    # Get dataset configs
    primary_cfg = schema.get_dataset_config(primary)
    secondary_cfg = schema.get_dataset_config(secondary)

    # Get join config
    try:
        join_info = schema.get_join_config(primary, secondary)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Join config lookup failed for %s + %s", primary, secondary)
        return _make_error_response(f"Join config error: {exc}", intent=intent)

    # Find the correct join key config
    join_config = join_info["config"]
    join_key_config = next(
        (jk for jk in join_config["join_keys"] if jk["type"] == join_type),
        None
    )
    if not join_key_config:
        return _make_error_response(
            f"Join type '{join_type}' not found in config.", intent=intent
        )

    # Adjust key order if join was defined in reverse
    if join_info["order"] == "reversed":
        join_key_config = {
            "left": join_key_config["right"],
            "right": join_key_config["left"],
            "type": join_key_config["type"],
        }

    # Load datasets
    primary_loader = LOADERS.get(primary)
    secondary_loader = LOADERS.get(secondary)
    if not primary_loader or not secondary_loader:
        return _make_error_response(
            f"Missing loader for '{primary}' or '{secondary}'.", intent=intent
        )

    # Default crime to 2024 when joining with vacant_properties (a current snapshot)
    _datasets = {primary, secondary}
    if "crime" in _datasets and "vacant_properties" in _datasets:
        filters = intent.get("filters") or {}
        if "year" not in filters:
            intent.setdefault("filters", {})["year"] = {"op": "=", "value": 2024}

    try:
        # Load data (cache with TTL for validation)
        primary_df = _get_cached_df(primary, primary_loader)
        secondary_df = _get_cached_df(secondary, secondary_loader)

        conn = duckdb.connect(database=":memory:")
        conn.register(primary_cfg["table"], primary_df)
        conn.register(secondary_cfg["table"], secondary_df)

        sql = build_join_sql(intent, primary_cfg, secondary_cfg, join_key_config)
        result_df = _execute_sql_with_timeout(conn, sql)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Join SQL execution failed")
        return _make_error_response(
            f"Failed to execute join SQL: {exc}",
            intent=intent,
            sql=sql if "sql" in locals() else None,
        )

    limitations = (
        f"Cross-dataset join via {join_type}. Static CSV snapshots only. "
        f"Join may exclude records with missing/null {join_type} values. "
        "If LLM was used, it was only for intent parsing; it never accesses data directly."
    )

    metadata = {
        "query_type": "join",
        "primary_dataset": primary,
        "secondary_dataset": secondary,
        "join_type": join_type,
        "join_description": join_config.get("description", ""),
        "filters": intent.get("filters") or {},
        "group_by": intent.get("group_by"),
        "metric": intent.get("metric", "count"),
        "limit": intent.get("limit"),
        "row_count": len(result_df),
    }

    # Run validation against ground-truth
    join_validation = validate_join_result(
        result_df, primary_df, secondary_df, intent, join_key_config
    )
    sanity_validation = sanity_check_result(result_df, intent, metadata)
    validation = combine_validations(join_validation, sanity_validation)

    return {
        "result": result_df,
        "raw_df": _df_cache.get(primary, (None,))[0],
        "intent": intent,
        "metadata": metadata,
        "limitations": limitations,
        "sql": sql,
        "error": None,
        "validation": validation.to_dict(),
        "bias_warnings": [],
    }


def _run_advanced_query(question: str) -> dict:
    """Execute a complex query using LLM-generated SQL with guardrails."""
    api_key = load_api_key()
    if not api_key:
        return _make_error_response(
            "Advanced SQL queries require an API key. Set OPENAI_API_KEY in .env."
        )

    # Generate SQL via LLM
    try:
        sql_llm = make_openai_sql_llm()
        prompt = NL_TO_SQL_PROMPT.format(question=question)
        raw_sql = sql_llm(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM SQL generation failed")
        return _make_error_response(f"Failed to generate SQL: {exc}")

    # Validate and sanitize SQL
    try:
        sql = validate_sql(raw_sql)
    except SQLValidationError as exc:
        return _make_error_response(
            f"Generated SQL failed safety checks: {exc}",
            sql=raw_sql,
        )

    # Load all tables and execute
    try:
        conn = duckdb.connect(database=":memory:")
        for name, loader in LOADERS.items():
            conn.register(name, _get_cached_df(name, loader))

        result_df = _execute_sql_with_timeout(conn, sql)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Advanced SQL execution failed")
        return _make_error_response(
            f"Failed to execute advanced SQL: {exc}", sql=sql
        )

    metadata = {
        "query_type": "advanced_sql",
        "row_count": len(result_df),
    }

    limitations = (
        "This query was generated by an LLM and executed via DuckDB. "
        "Results should be independently verified. "
        "Static CSV snapshots only."
    )

    return {
        "result": result_df,
        "intent": {"query_path": "advanced_sql", "question": question},
        "metadata": metadata,
        "limitations": limitations,
        "sql": sql,
        "error": None,
        "validation": {"passed": True, "warnings": ["LLM-generated SQL — no ground-truth validation"], "errors": [], "ground_truth": {}},
        "bias_warnings": [],
    }


def main() -> int:
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not question:
        try:
            question = input("Ask Syracuse Data > ").strip()
        except KeyboardInterrupt:
            return 0

    response = run_query(question)

    print("\n--- Result ---")
    if isinstance(response.get("result"), pd.DataFrame):
        print(response["result"].to_string(index=False))
    else:
        print("No result. Reason:", response.get("error") or response.get("limitations"))

    print("\n--- SQL ---")
    print(response.get("sql", "N/A"))

    print("\n--- Metadata ---")
    for k, v in (response.get("metadata") or {}).items():
        print(f"{k}: {v}")

    print("\n--- Limitations ---")
    print(response.get("limitations"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
