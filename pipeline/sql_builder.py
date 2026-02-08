"""
SQL builder for Ask Syracuse Data.
Takes a validated intent and dataset config, returns DuckDB-compatible SELECT SQL.
Supports: count, count_distinct, avg/min/max on computed columns,
temporal GROUP BY, richer WHERE filters, HAVING, multiple GROUP BY.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional


def _quote(value: str) -> str:
    """Safely quote a string value for SQL."""
    return "'" + value.replace("'", "''") + "'"


# =============================================================================
# COMPOSABLE SQL HELPERS
# =============================================================================

def _build_metric_expr(intent: Dict[str, Any], config: Dict[str, Any]) -> tuple:
    """
    Build the metric SELECT expression and its alias.

    Returns:
        (sql_expr, alias) e.g. ("COUNT(*)", "count")
    """
    metric = intent.get("metric", "count")

    if metric == "count":
        return "COUNT(*)", "count"

    if metric == "count_distinct":
        distinct_col = intent.get("distinct_column", "sbl")
        return f"COUNT(DISTINCT {distinct_col})", "count_distinct"

    if metric in ("avg", "min", "max", "sum"):
        metric_column = intent["metric_column"]
        computed = config.get("computed_columns", {}).get(metric_column, {})
        expr = computed.get("expr", metric_column)
        alias = f"{metric}_{metric_column}"
        return f"{metric.upper()}({expr})", alias

    # Fallback
    return "COUNT(*)", "count"


def _build_group_exprs(
    group_by: Optional[List[str]], config: Dict[str, Any]
) -> tuple:
    """
    Build GROUP BY SELECT expressions and GROUP BY clauses.
    Handles temporal groups (year, month, quarter) via date_part().

    Returns:
        (select_exprs, group_by_exprs) — lists of SQL fragments
    """
    if not group_by:
        return [], []

    temporal_map = config.get("temporal_group_map", {})
    select_exprs = []
    group_exprs = []

    for gb in group_by:
        if gb in temporal_map:
            date_col, part = temporal_map[gb]
            expr = f"date_part('{part}', {date_col})"
            select_exprs.append(f"{expr} AS {gb}")
            group_exprs.append(expr)
        else:
            select_exprs.append(gb)
            group_exprs.append(gb)

    return select_exprs, group_exprs


def _filter_conditions(
    filters: Dict[str, Any], config: Dict[str, Any], col_prefix: str = ""
) -> List[str]:
    """
    Build a list of SQL WHERE conditions from normalized filters.

    Filters are in format: {"key": {"op": "=", "value": X}}
    Handles: =, >=, <=, >, <, between, in, like
    Special: "year" filter uses date_part() on the date_column.

    col_prefix: optional table alias prefix like "p." for join queries.
    """
    if not filters:
        return []

    date_col = config.get("date_column")
    temporal_map = config.get("temporal_group_map", {})
    clauses = []

    for key, filt in filters.items():
        op = filt["op"]
        value = filt["value"]

        # Determine the SQL expression for the column
        if key == "year":
            if not date_col:
                raise ValueError("Dataset has no date column for year filter.")
            col_expr = f"date_part('year', {col_prefix}{date_col})"
        elif key in temporal_map:
            tc, part = temporal_map[key]
            col_expr = f"date_part('{part}', {col_prefix}{tc})"
        else:
            col_expr = f"{col_prefix}{key}"

        # Build the condition based on operator
        if op == "=":
            if isinstance(value, (int, float)):
                clauses.append(f"{col_expr} = {value}")
            else:
                clauses.append(f"{col_expr} = {_quote(str(value))}")
        elif op in (">=", "<=", ">", "<"):
            if isinstance(value, (int, float)):
                clauses.append(f"{col_expr} {op} {value}")
            else:
                clauses.append(f"{col_expr} {op} {_quote(str(value))}")
        elif op == "between":
            clauses.append(f"{col_expr} BETWEEN {value[0]} AND {value[1]}")
        elif op == "in":
            if all(isinstance(v, (int, float)) for v in value):
                vals = ", ".join(str(v) for v in value)
            else:
                vals = ", ".join(_quote(str(v)) for v in value)
            clauses.append(f"{col_expr} IN ({vals})")
        elif op == "like":
            clauses.append(f"{col_expr} LIKE {_quote(str(value))}")

    return clauses


def _build_where_clause(
    filters: Dict[str, Any], config: Dict[str, Any], col_prefix: str = ""
) -> str:
    """Build complete WHERE clause string from filters."""
    conditions = _filter_conditions(filters, config, col_prefix)
    if not conditions:
        return ""
    return " WHERE " + " AND ".join(conditions)


def _build_null_filter(intent: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Build null-filter for computed column metrics (avg/min/max).
    These need to exclude rows where source date columns are NULL."""
    metric = intent.get("metric", "count")
    if metric not in ("avg", "min", "max", "sum"):
        return ""
    metric_column = intent.get("metric_column")
    if not metric_column:
        return ""
    computed = config.get("computed_columns", {}).get(metric_column, {})
    return computed.get("null_filter", "")


def _build_having_clause(
    having: Optional[Dict[str, Any]], metric_expr: str
) -> str:
    """Build HAVING clause from having dict."""
    if not having:
        return ""
    return f" HAVING {metric_expr} {having['op']} {having['value']}"


# =============================================================================
# SINGLE-DATASET QUERIES
# =============================================================================

def build_select_sql(intent: Dict[str, Any], dataset_config: Dict[str, Any]) -> str:
    """
    Build a single-dataset SELECT query from a validated intent.

    Supports:
    - count, count_distinct, avg/min/max on computed columns
    - Temporal GROUP BY (year, month, quarter via date_part)
    - Multiple GROUP BY columns
    - Richer WHERE filters (=, >=, <=, between, in, like)
    - HAVING clause
    """
    table = dataset_config["table"]
    group_by = intent.get("group_by")  # list or None
    filters = intent.get("filters") or {}
    limit = intent.get("limit")
    having = intent.get("having")

    # Build metric expression
    metric_expr_raw, metric_alias = _build_metric_expr(intent, dataset_config)

    # Build GROUP BY expressions
    group_select, group_by_exprs = _build_group_exprs(group_by, dataset_config)

    # SELECT clause
    select_parts = group_select + [f"{metric_expr_raw} AS {metric_alias}"]
    select_clause = ", ".join(select_parts)

    # WHERE clause
    where_clause = _build_where_clause(filters, dataset_config)

    # Add null filter for computed column metrics
    null_filter = _build_null_filter(intent, dataset_config)
    if null_filter:
        if where_clause:
            where_clause += f" AND {null_filter}"
        else:
            where_clause = f" WHERE {null_filter}"

    # GROUP BY clause
    group_clause = ""
    if group_by_exprs:
        group_clause = " GROUP BY " + ", ".join(group_by_exprs)

    # HAVING clause (only valid with GROUP BY)
    having_clause = ""
    if group_by_exprs and having:
        having_clause = _build_having_clause(having, metric_expr_raw)

    # ORDER BY clause
    if group_by_exprs:
        temporal_map = dataset_config.get("temporal_group_map", {})
        temporal_groups = [gb for gb in (group_by or []) if gb in temporal_map]
        if temporal_groups and len(group_by) == 1:
            # Single temporal grouping: order by time ascending
            order_clause = f" ORDER BY {group_by[0]}"
        else:
            order_clause = f" ORDER BY {metric_alias} DESC"
    else:
        order_clause = ""

    # LIMIT clause
    limit_clause = f" LIMIT {limit}" if limit else ""

    return (
        f"SELECT {select_clause} FROM {table}"
        f"{where_clause}{group_clause}{having_clause}{order_clause}{limit_clause};"
    )


# =============================================================================
# CROSS-DATASET JOIN QUERIES
# =============================================================================

def _build_zip_join_sql(
    intent: Dict[str, Any],
    primary_config: Dict[str, Any],
    secondary_config: Dict[str, Any],
    join_key_config: Dict[str, Any],
) -> str:
    """
    Build a ZIP-based JOIN using CTEs to pre-aggregate each dataset.
    Avoids many-to-many inflation by aggregating each dataset by ZIP first.
    """
    primary_table = primary_config["table"]
    secondary_table = secondary_config["table"]
    left_key = join_key_config["left"]
    right_key = join_key_config["right"]
    filters = intent.get("filters") or {}
    limit = intent.get("limit")

    # WHERE clause for primary CTE (no alias prefix)
    primary_where = _build_where_clause(filters, primary_config)
    limit_clause = f" LIMIT {limit}" if limit else ""

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
    SBL is a property-level identifier, so direct joins are appropriate.
    """
    primary_table = primary_config["table"]
    secondary_table = secondary_config["table"]
    left_key = join_key_config["left"]
    right_key = join_key_config["right"]
    group_by = intent.get("group_by")
    filters = intent.get("filters") or {}
    limit = intent.get("limit")

    p_alias = "p"
    s_alias = "s"

    # Build SELECT clause
    select_parts = []
    if group_by:
        select_parts.append(f"{p_alias}.{group_by}")
    select_parts.append(f"COUNT({s_alias}.*) AS count")
    select_clause = ", ".join(select_parts)

    # Build JOIN clause
    join_clause = (
        f"LEFT JOIN {secondary_table} {s_alias} "
        f"ON {p_alias}.{left_key} = {s_alias}.{right_key}"
    )

    # Build WHERE clause (with alias prefix)
    where_clause = _build_where_clause(filters, primary_config, col_prefix=f"{p_alias}.")

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


def _build_neighborhood_join_sql(
    intent: Dict[str, Any],
    primary_config: Dict[str, Any],
    secondary_config: Dict[str, Any],
    join_key_config: Dict[str, Any],
) -> str:
    """
    Build a neighborhood-based JOIN using CTEs to pre-aggregate each dataset.
    Same CTE pattern as ZIP joins to avoid many-to-many inflation.
    """
    primary_table = primary_config["table"]
    secondary_table = secondary_config["table"]
    left_key = join_key_config["left"]
    right_key = join_key_config["right"]
    filters = intent.get("filters") or {}
    limit = intent.get("limit")

    # WHERE clause for primary CTE (no alias prefix)
    primary_where = _build_where_clause(filters, primary_config)
    limit_clause = f" LIMIT {limit}" if limit else ""

    sql = f"""WITH primary_by_nh AS (
    SELECT {left_key} AS neighborhood, COUNT(*) AS primary_count
    FROM {primary_table}{primary_where}
    GROUP BY {left_key}
),
secondary_by_nh AS (
    SELECT {right_key} AS neighborhood, COUNT(*) AS secondary_count
    FROM {secondary_table}
    GROUP BY {right_key}
)
SELECT
    p.neighborhood,
    p.primary_count,
    COALESCE(s.secondary_count, 0) AS secondary_count
FROM primary_by_nh p
LEFT JOIN secondary_by_nh s ON p.neighborhood = s.neighborhood
ORDER BY p.primary_count DESC{limit_clause};"""

    return sql


def build_join_sql(
    intent: Dict[str, Any],
    primary_config: Dict[str, Any],
    secondary_config: Dict[str, Any],
    join_key_config: Dict[str, Any],
) -> str:
    """
    Build a JOIN SQL query for cross-dataset analysis.

    Routes to the appropriate join builder based on join_type:
    - ZIP: CTE-based pre-aggregation
    - Neighborhood: CTE-based pre-aggregation
    - SBL: Direct LEFT JOIN for property-level analysis
    """
    join_type = join_key_config.get("type", "sbl")

    if join_type == "zip":
        return _build_zip_join_sql(intent, primary_config, secondary_config, join_key_config)
    elif join_type == "neighborhood":
        return _build_neighborhood_join_sql(intent, primary_config, secondary_config, join_key_config)
    else:
        return _build_sbl_join_sql(intent, primary_config, secondary_config, join_key_config)


__all__ = ["build_select_sql", "build_join_sql"]
