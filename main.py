"""
CLI entry point for Ask Syracuse Data.
Pipeline: question -> LLM intent -> validate -> SQL -> DuckDB -> result.
"""
from __future__ import annotations
import sys
import duckdb
import pandas as pd

import schema
from data_utils import (
    load_code_violations,
    load_rental_registry,
    load_vacant_properties,
    load_crime_2022,
)
from llm.intent_parser import parse_intent
from llm.openai_client import make_openai_intent_llm, load_api_key
from sql_builder import build_select_sql


LOADERS = {
    "violations": load_code_violations,
    "vacant_properties": load_vacant_properties,
    "crime_2022": load_crime_2022,
    "rental_registry": load_rental_registry,
}


def run_query(question: str) -> dict:
    """Run the full pipeline and return a structured response."""
    llm_fn = None
    api_key = load_api_key()
    if api_key:
        try:
            llm_fn = make_openai_intent_llm()
        except Exception as exc:  # noqa: BLE001
            return {
                "result": None,
                "intent": None,
                "metadata": {},
                "sql": None,
                "error": f"Unable to initialize LLM client: {exc}",
                "limitations": "",
            }

    try:
        raw_intent = parse_intent(question, llm=llm_fn)
        intent = schema.validate_intent(raw_intent)
    except Exception as exc:  # noqa: BLE001
        note = " (GPT disabled; set OPENAI_API_KEY to enable)" if llm_fn is None else ""
        return {
            "result": None,
            "intent": raw_intent if "raw_intent" in locals() else None,
            "metadata": {},
            "sql": None,
            "error": f"Unable to parse/validate intent: {exc}{note}",
            "limitations": "",
        }

    dataset = intent["dataset"]
    cfg = schema.get_dataset_config(dataset)
    loader = LOADERS.get(dataset)
    if not loader:
        return {
            "result": None,
            "intent": intent,
            "metadata": {},
            "sql": None,
            "error": f"No loader found for dataset '{dataset}'.",
            "limitations": "",
        }

    try:
        df = loader()
        conn = duckdb.connect(database=":memory:")
        conn.register(cfg["table"], df)
        sql = build_select_sql(intent, cfg)
        result_df = conn.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        return {
            "result": None,
            "intent": intent,
            "metadata": {"intent": intent},
            "sql": sql if "sql" in locals() else None,
            "error": f"Failed to execute SQL: {exc}",
            "limitations": "",
        }

    limitations = (
        "Static CSV snapshots only; DuckDB queries are deterministic. "
        "LLM used only for intent parsing; it never accesses data."
    )

    metadata = {
        "dataset": dataset,
        "filters": intent.get("filters") or {},
        "group_by": intent.get("group_by"),
        "metric": intent.get("metric"),
        "limit": intent.get("limit"),
        "row_count": len(result_df),
    }

    return {
        "result": result_df,
        "intent": intent,
        "metadata": metadata,
        "limitations": limitations,
        "sql": sql,
        "error": None,
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
