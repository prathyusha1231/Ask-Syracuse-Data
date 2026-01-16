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


def _build_zip_join_sql(
    intent: Dict[str, Any],
    primary_config: Dict[str, Any],
    secondary_config: Dict[str, Any],
    join_key_config: Dict[str, Any],
) -> str:
    """
    Build a ZIP-based JOIN using CTEs to pre-aggregate each dataset.

    This avoids the many-to-many inflation problem by aggregating each
    dataset by ZIP first, then joining the summaries.
    """
    primary_table = primary_config["table"]
    secondary_table = secondary_config["table"]

    left_key = join_key_config["left"]   # ZIP field in primary table
    right_key = join_key_config["right"]  # ZIP field in secondary table

    filters = intent.get("filters") or {}
    limit = intent.get("limit")
    date_col = primary_config.get("date_column")

    # Build WHERE clause for primary table
    where_clauses = []
    for key, value in filters.items():
        if key == "year":
            if not date_col:
                raise ValueError("Primary dataset has no date column for year filter.")
            where_clauses.append(f"date_part('year', {date_col}) = {int(value)}")
        else:
            where_clauses.append(f"{key} = {_quote(str(value))}")

    primary_where = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Build LIMIT clause
    limit_clause = f" LIMIT {limit}" if limit else ""

    # CTE-based query: aggregate each table by ZIP, then join
    sql = f"""WITH primary_by_zip AS (
    SELECT {left_key} AS zip, COUNT(*) AS primary_count
    FROM {primary_table}{primary_where}
    GROUP BY {left_key}
),
secondary_by_zip AS (
    SELECT {right_key} AS zip, COUNT(*) AS secondary_count
    FROM {secondary_table}
    GROUP BY {right_key}
)
SELECT
    p.zip,
    p.primary_count,
    COALESCE(s.secondary_count, 0) AS secondary_count
FROM primary_by_zip p
LEFT JOIN secondary_by_zip s ON p.zip = s.zip
ORDER BY p.primary_count DESC{limit_clause};"""

    return sql


def _build_sbl_join_sql(
    intent: Dict[str, Any],
    primary_config: Dict[str, Any],
    secondary_config: Dict[str, Any],
    join_key_config: Dict[str, Any],
) -> str:
    """
    Build an SBL-based JOIN using direct LEFT JOIN.

    SBL (Standard Boundary Locator) is a property-level identifier,
    so direct joins are appropriate without pre-aggregation.
    """
    primary_table = primary_config["table"]
    secondary_table = secondary_config["table"]

    left_key = join_key_config["left"]
    right_key = join_key_config["right"]

    group_by = intent.get("group_by")
    filters = intent.get("filters") or {}
    limit = intent.get("limit")
    date_col = primary_config.get("date_column")

    # Use table aliases for clarity
    p_alias = "p"  # primary
    s_alias = "s"  # secondary

    # Build SELECT clause
    select_parts = []
    if group_by:
        select_parts.append(f"{p_alias}.{group_by}")
    select_parts.append(f"COUNT({s_alias}.*) AS count")
    select_clause = ", ".join(select_parts)

    # Build JOIN clause
    join_clause = f"LEFT JOIN {secondary_table} {s_alias} ON {p_alias}.{left_key} = {s_alias}.{right_key}"

    # Build WHERE clause (filters apply to primary table)
    where_clauses = []
    for key, value in filters.items():
        if key == "year":
            if not date_col:
                raise ValueError("Primary dataset has no date column for year filter.")
            where_clauses.append(f"date_part('year', {p_alias}.{date_col}) = {int(value)}")
        else:
            where_clauses.append(f"{p_alias}.{key} = {_quote(str(value))}")

    where_clause = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Build GROUP BY clause
    group_clause = f" GROUP BY {p_alias}.{group_by}" if group_by else ""

    # Build ORDER BY clause
    order_clause = " ORDER BY count DESC" if group_by else ""

    # Build LIMIT clause
    limit_clause = f" LIMIT {limit}" if limit else ""

    sql = (
        f"SELECT {select_clause} "
        f"FROM {primary_table} {p_alias} "
        f"{join_clause}"
        f"{where_clause}"
        f"{group_clause}"
        f"{order_clause}"
        f"{limit_clause};"
    )
    return sql


def build_join_sql(
    intent: Dict[str, Any],
    primary_config: Dict[str, Any],
    secondary_config: Dict[str, Any],
    join_key_config: Dict[str, Any],
) -> str:
    """
    Build a JOIN SQL query for cross-dataset analysis.

    Args:
        intent: Validated join intent with primary_dataset, secondary_dataset, join_type, etc.
        primary_config: Dataset config for primary table
        secondary_config: Dataset config for secondary table
        join_key_config: Join key info with 'left', 'right' field names and 'type'

    Returns:
        DuckDB-compatible SQL string

    Notes:
        - ZIP-based joins use CTEs to pre-aggregate and avoid many-to-many inflation
        - SBL-based joins use direct LEFT JOIN for property-level analysis
    """
    join_type = join_key_config.get("type", "sbl")

    if join_type == "zip":
        return _build_zip_join_sql(intent, primary_config, secondary_config, join_key_config)
    else:
        return _build_sbl_join_sql(intent, primary_config, secondary_config, join_key_config)


__all__ = ["build_select_sql", "build_join_sql"]
