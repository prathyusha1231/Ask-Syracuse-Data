"""
SQL validator/guardrails for LLM-generated SQL in Ask Syracuse Data.
Ensures LLM-generated SQL is safe, read-only, and scoped to allowed tables.
"""
from __future__ import annotations
import re
from typing import Set


class SQLValidationError(Exception):
    """Raised when SQL fails guardrail validation."""


# Allowed tables
ALLOWED_TABLES: Set[str] = {
    "violations",
    "rental_registry",
    "vacant_properties",
    "crime",
    "unfit_properties",
    "trash_pickup",
    "historical_properties",
    "assessment_roll",
    "cityline_requests",
    "snow_routes",
    "bike_suitability",
    "bike_infrastructure",
    "parking_violations",
    "permit_requests",
    "tree_inventory",
    "lead_testing",
}

# Dangerous SQL keywords (write operations)
WRITE_OPS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "MERGE", "REPLACE",
}

# Dangerous functions/features
DANGEROUS_PATTERNS = [
    r"\bread_csv\b",
    r"\bread_parquet\b",
    r"\bhttpfs\b",
    r"\bhttp_get\b",
    r"\bcopy\b",
    r"\bexport\b",
    r"\binformation_schema\b",
    r"\bpg_catalog\b",
    r"\bpg_\w+",
    r"\bsystem\b",
    r"\bexec\b",
    r"\beval\b",
    r"\bload_extension\b",
    r"\binstall\b",
    r"\battach\b",
    r"\bdetach\b",
]

MAX_LIMIT = 1000


def _strip_markdown_fences(sql: str) -> str:
    """Remove markdown code fences from LLM output."""
    sql = sql.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


def _extract_cte_names(sql: str) -> Set[str]:
    """Extract CTE (Common Table Expression) alias names from WITH clauses."""
    cte_names = set()
    if not sql.strip().upper().startswith("WITH"):
        return cte_names
    # Match: WITH name AS ( and , name AS (
    # First CTE after WITH
    m = re.match(r"WITH\s+(\w+)\s+AS\s*\(", sql, re.IGNORECASE)
    if m:
        cte_names.add(m.group(1).lower())
    # Subsequent CTEs after ),
    for m in re.finditer(r"\)\s*,\s*(\w+)\s+AS\s*\(", sql, re.IGNORECASE):
        cte_names.add(m.group(1).lower())
    return cte_names


def _extract_table_names(sql: str) -> Set[str]:
    """Extract table names referenced in FROM and JOIN clauses, excluding CTE aliases."""
    tables = set()

    # Match FROM table_name and JOIN table_name patterns
    pattern = r"(?:FROM|JOIN)\s+(\w+)"
    for match in re.finditer(pattern, sql, re.IGNORECASE):
        tables.add(match.group(1).lower())

    # Remove CTE aliases — they're not real tables
    cte_names = _extract_cte_names(sql)
    tables -= cte_names

    return tables


def _has_limit(sql: str) -> bool:
    """Check if SQL has a LIMIT clause."""
    return bool(re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE))


def _inject_limit(sql: str, limit: int = MAX_LIMIT) -> str:
    """Inject a LIMIT clause if not present."""
    sql = sql.rstrip().rstrip(";")

    # Check for existing LIMIT
    limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
    if limit_match:
        existing_limit = int(limit_match.group(1))
        if existing_limit > limit:
            # Replace with max allowed
            sql = re.sub(
                r"\bLIMIT\s+\d+",
                f"LIMIT {limit}",
                sql,
                flags=re.IGNORECASE,
            )
        return sql + ";"

    # No LIMIT found — append one
    return sql + f" LIMIT {limit};"


def validate_sql(sql: str) -> str:
    """
    Validate and sanitize LLM-generated SQL.

    Checks:
    1. Must start with SELECT or WITH (CTEs allowed)
    2. No write operations (INSERT, UPDATE, DELETE, etc.)
    3. No dangerous functions (read_csv, httpfs, etc.)
    4. Only allowed tables
    5. LIMIT enforcement (inject if missing, cap at 1000)

    Returns:
        Sanitized SQL string

    Raises:
        SQLValidationError if SQL fails validation
    """
    # Strip markdown fences
    sql = _strip_markdown_fences(sql)

    if not sql:
        raise SQLValidationError("Empty SQL query.")

    # Normalize whitespace for checks
    sql_upper = sql.upper().strip()

    # Must start with SELECT or WITH
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        raise SQLValidationError(
            "SQL must start with SELECT or WITH. "
            f"Got: {sql[:30]}..."
        )

    # Check for write operations
    for op in WRITE_OPS:
        # Match as whole word to avoid false positives (e.g., "UPDATED" column name)
        if re.search(rf"\b{op}\b", sql, re.IGNORECASE):
            raise SQLValidationError(f"Write operation '{op}' is not allowed.")

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE):
            raise SQLValidationError(
                f"Dangerous pattern detected: {pattern}. External data access is not allowed."
            )

    # Check table names
    tables = _extract_table_names(sql)
    invalid_tables = tables - ALLOWED_TABLES
    if invalid_tables:
        raise SQLValidationError(
            f"Unknown table(s): {', '.join(sorted(invalid_tables))}. "
            f"Allowed: {', '.join(sorted(ALLOWED_TABLES))}"
        )

    if not tables:
        raise SQLValidationError("No tables detected in query.")

    # Enforce LIMIT
    sql = _inject_limit(sql)

    return sql


__all__ = ["validate_sql", "SQLValidationError"]
