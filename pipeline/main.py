"""
CLI entry point for Ask Syracuse Data.
Pipeline: question -> LLM intent -> validate -> SQL -> DuckDB -> result.
Supports both single-dataset queries and cross-dataset joins.
Includes validation against ground-truth and bias detection.
"""
from __future__ import annotations
import sys
import duckdb
import pandas as pd

from . import schema
from .data_utils import (
    load_code_violations,
    load_rental_registry,
    load_vacant_properties,
    load_crime_2022,
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
)
from llm.intent_parser import parse_intent
from llm.openai_client import make_openai_intent_llm, make_openai_sql_llm, load_api_key
from llm.prompt_templates import NL_TO_SQL_PROMPT
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
    "crime_2022": load_crime_2022,
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
_df_cache = {}


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


def run_query(question: str) -> dict:
    """Run the full pipeline and return a structured response."""
    llm_fn = None
    api_key = load_api_key()
    if api_key:
        try:
            llm_fn = make_openai_intent_llm()
        except Exception as exc:  # noqa: BLE001
            return _make_error_response(f"Unable to initialize LLM client: {exc}")

    raw_intent = None
    try:
        raw_intent = parse_intent(question, llm=llm_fn)
    except Exception as exc:  # noqa: BLE001
        note = " (GPT disabled; set OPENAI_API_KEY to enable)" if llm_fn is None else ""
        return _make_error_response(f"Unable to parse intent: {exc}{note}", intent=raw_intent)

    # Route to the appropriate query handler
    if raw_intent.get("query_path") == "advanced_sql":
        response = _run_advanced_query(raw_intent["question"])
    elif raw_intent.get("query_type") == "join":
        response = _run_join_query(raw_intent)
    else:
        response = _run_single_query(raw_intent)

    # Run bias detection on successful queries
    if response.get("result") is not None and response.get("error") is None:
        bias_result = run_all_bias_checks(
            question,
            response["result"],
            response.get("intent", {}),
            response.get("metadata", {}),
        )
        response["bias_warnings"] = bias_result.to_list()

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
        return _make_error_response(f"Validation failed: {exc}", intent=raw_intent)

    dataset = intent["dataset"]
    cfg = schema.get_dataset_config(dataset)
    loader = LOADERS.get(dataset)
    if not loader:
        return _make_error_response(
            f"No loader found for dataset '{dataset}'.", intent=intent
        )

    try:
        # Load data (cache for validation)
        if dataset not in _df_cache:
            _df_cache[dataset] = loader()
        df = _df_cache[dataset]

        conn = duckdb.connect(database=":memory:")
        conn.register(cfg["table"], df)
        sql = build_select_sql(intent, cfg)
        result_df = conn.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        return _make_error_response(
            f"Failed to execute SQL: {exc}",
            intent=intent,
            metadata={"intent": intent},
            sql=sql if "sql" in locals() else None,
        )

    limitations = (
        "Static CSV snapshots only; DuckDB queries are deterministic. "
        "LLM used only for intent parsing; it never accesses data."
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

    try:
        # Load data (cache for validation)
        if primary not in _df_cache:
            _df_cache[primary] = primary_loader()
        if secondary not in _df_cache:
            _df_cache[secondary] = secondary_loader()

        primary_df = _df_cache[primary]
        secondary_df = _df_cache[secondary]

        conn = duckdb.connect(database=":memory:")
        conn.register(primary_cfg["table"], primary_df)
        conn.register(secondary_cfg["table"], secondary_df)

        sql = build_join_sql(intent, primary_cfg, secondary_cfg, join_key_config)
        result_df = conn.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        return _make_error_response(
            f"Failed to execute join SQL: {exc}",
            intent=intent,
            sql=sql if "sql" in locals() else None,
        )

    limitations = (
        f"Cross-dataset join via {join_type}. Static CSV snapshots only. "
        f"Join may exclude records with missing/null {join_type} values. "
        "LLM used only for intent parsing; it never accesses data."
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
            if name not in _df_cache:
                _df_cache[name] = loader()
            conn.register(name, _df_cache[name])

        result_df = conn.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
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
