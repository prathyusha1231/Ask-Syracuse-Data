"""
SQL builder for Ask Syracuse Data.
Takes a validated intent and dataset config, returns DuckDB-compatible SELECT SQL.
"""
from __future__ import annotations
from typing import Dict, Any


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_select_sql(intent: Dict[str, Any], dataset_config: Dict[str, Any]) -> str:
    dataset = intent["dataset"]
    table = dataset_config["table"]
    group_by = intent.get("group_by")
    filters = intent.get("filters") or {}
    limit = intent.get("limit")
    date_col = dataset_config.get("date_column")

    select_parts = []
    if group_by:
        select_parts.append(group_by)
    select_parts.append("count(*) AS count")
    select_clause = ", ".join(select_parts)

    where_clauses = []
    for key, value in filters.items():
        if key == "year":
            if not date_col:
                raise ValueError(f"Dataset '{dataset}' has no date column for year filter.")
            where_clauses.append(f"date_part('year', {date_col}) = {int(value)}")
        else:
            where_clauses.append(f"{key} = {_quote(str(value))}")

    where_clause = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    group_clause = f" GROUP BY {group_by}" if group_by else ""
    order_clause = " ORDER BY count DESC" if group_by else ""
    limit_clause = f" LIMIT {limit}" if limit else ""

    sql = f"SELECT {select_clause} FROM {table}{where_clause}{group_clause}{order_clause}{limit_clause};"
    return sql


__all__ = ["build_select_sql"]
