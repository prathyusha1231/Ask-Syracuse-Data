"""
Data Quality Module for Ask Syracuse Data.
Handles null values, data validation, and quality reporting.
"""
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Callable
from data_utils import (
    load_code_violations,
    load_rental_registry,
    load_vacant_properties,
    load_crime_2022,
)


# =============================================================================
# NULL HANDLING STRATEGIES
# =============================================================================
# For each dataset and column, define how to handle nulls:
# - "drop": Remove rows with nulls (use sparingly)
# - "label": Replace with a descriptive label like "Not Recorded"
# - "impute_mode": Fill with most common value (for categorical)
# - "impute_median": Fill with median (for numeric)
# - "impute_by_group": Fill based on another column's grouping
# - "keep": Leave as-is (for optional fields)

NULL_STRATEGIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "violations": {
        "neighborhood": {
            "strategy": "label",
            "label": "Not Recorded",
            "reason": "94 properties (likely commercial/institutional) lack neighborhood assignment"
        },
        "complaint_zip": {
            "strategy": "drop",
            "reason": "ZIP is essential for geographic analysis"
        },
        "violation": {
            "strategy": "label",
            "label": "Unspecified Violation",
            "reason": "Violation type should be recorded but may be missing"
        },
        "status_type_name": {
            "strategy": "label",
            "label": "Unknown Status",
            "reason": "Status should be recorded"
        },
        "sbl": {
            "strategy": "keep",
            "reason": "SBL may be missing for some records, still usable for ZIP analysis"
        },
    },
    "rental_registry": {
        "zip": {
            "strategy": "drop",
            "reason": "ZIP is essential for analysis"
        },
        "sbl": {
            "strategy": "keep",
            "reason": "SBL needed for property-level joins"
        },
        "completion_type_name": {
            "strategy": "label",
            "label": "Not Recorded",
            "reason": "Completion type may not be set"
        },
    },
    "vacant_properties": {
        "neighborhood": {
            "strategy": "label",
            "label": "Not Recorded",
            "reason": "Some properties may lack neighborhood assignment"
        },
        "zip": {
            "strategy": "drop",
            "reason": "ZIP is essential for analysis"
        },
        "sbl": {
            "strategy": "keep",
            "reason": "SBL needed for property-level joins"
        },
    },
    "crime_2022": {
        "code_defined": {
            "strategy": "label",
            "label": "Unspecified Crime",
            "reason": "Crime type should be recorded"
        },
        "arrest": {
            "strategy": "label",
            "label": "Unknown",
            "reason": "Arrest status may not be recorded"
        },
    },
}


# =============================================================================
# DATA QUALITY AUDIT
# =============================================================================
def audit_dataset(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Audit a dataset for null values and data quality issues.

    Returns:
        Dict with null counts, percentages, and recommendations
    """
    total_rows = len(df)
    audit_results = {
        "dataset": dataset_name,
        "total_rows": total_rows,
        "columns": {},
        "overall_completeness": 0.0,
    }

    total_cells = 0
    null_cells = 0

    for col in df.columns:
        col_nulls = df[col].isna().sum()
        # Also check for empty strings
        if df[col].dtype == 'object':
            col_nulls += (df[col] == '').sum()

        null_pct = (col_nulls / total_rows * 100) if total_rows > 0 else 0

        # Get strategy if defined
        strategy_info = NULL_STRATEGIES.get(dataset_name, {}).get(col, {})

        audit_results["columns"][col] = {
            "null_count": int(col_nulls),
            "null_percentage": round(null_pct, 2),
            "strategy": strategy_info.get("strategy", "keep"),
            "reason": strategy_info.get("reason", "No strategy defined"),
        }

        total_cells += total_rows
        null_cells += col_nulls

    audit_results["overall_completeness"] = round(
        (1 - null_cells / total_cells) * 100 if total_cells > 0 else 100, 2
    )

    return audit_results


def audit_all_datasets() -> Dict[str, Any]:
    """Audit all datasets and return comprehensive quality report."""
    loaders = {
        "violations": load_code_violations,
        "rental_registry": load_rental_registry,
        "vacant_properties": load_vacant_properties,
        "crime_2022": load_crime_2022,
    }

    report = {"datasets": {}, "summary": {}}

    for name, loader in loaders.items():
        try:
            df = loader()
            report["datasets"][name] = audit_dataset(df, name)
        except Exception as e:
            report["datasets"][name] = {"error": str(e)}

    # Summary statistics
    total_rows = sum(
        d.get("total_rows", 0)
        for d in report["datasets"].values()
        if "total_rows" in d
    )
    avg_completeness = sum(
        d.get("overall_completeness", 0)
        for d in report["datasets"].values()
        if "overall_completeness" in d
    ) / len(report["datasets"])

    report["summary"] = {
        "total_datasets": len(loaders),
        "total_rows": total_rows,
        "average_completeness": round(avg_completeness, 2),
    }

    return report


# =============================================================================
# NULL HANDLING FUNCTIONS
# =============================================================================
def apply_null_strategy(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Apply null handling strategies to a dataset.

    Args:
        df: The DataFrame to clean
        dataset_name: Name of the dataset (must be in NULL_STRATEGIES)

    Returns:
        Cleaned DataFrame with nulls handled according to strategy
    """
    df = df.copy()
    strategies = NULL_STRATEGIES.get(dataset_name, {})

    for col, config in strategies.items():
        if col not in df.columns:
            continue

        strategy = config.get("strategy", "keep")

        if strategy == "drop":
            # Drop rows where this column is null
            df = df.dropna(subset=[col])
            # Also drop empty strings for object columns
            if df[col].dtype == 'object':
                df = df[df[col] != '']

        elif strategy == "label":
            label = config.get("label", "Not Recorded")
            # Fill nulls with label
            df[col] = df[col].fillna(label)
            # Also replace empty strings
            if df[col].dtype == 'object':
                df[col] = df[col].replace('', label)

        elif strategy == "impute_mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

        elif strategy == "impute_median":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        # "keep" strategy - do nothing

    return df


def get_null_summary(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Get detailed summary of null values in a specific column."""
    null_mask = df[col].isna()
    if df[col].dtype == 'object':
        null_mask = null_mask | (df[col] == '')

    null_count = null_mask.sum()
    total = len(df)

    return {
        "column": col,
        "null_count": int(null_count),
        "total_rows": total,
        "null_percentage": round(null_count / total * 100, 2) if total > 0 else 0,
        "non_null_count": int(total - null_count),
    }


# =============================================================================
# QUALITY REPORT FORMATTING
# =============================================================================
def format_quality_report(report: Dict[str, Any]) -> str:
    """Format quality report as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("DATA QUALITY REPORT")
    lines.append("=" * 60)
    lines.append("")

    summary = report.get("summary", {})
    lines.append(f"Total Datasets: {summary.get('total_datasets', 0)}")
    lines.append(f"Total Records: {summary.get('total_rows', 0):,}")
    lines.append(f"Average Completeness: {summary.get('average_completeness', 0)}%")
    lines.append("")

    for dataset_name, dataset_info in report.get("datasets", {}).items():
        if "error" in dataset_info:
            lines.append(f"\n{dataset_name.upper()}: Error - {dataset_info['error']}")
            continue

        lines.append("-" * 60)
        lines.append(f"{dataset_name.upper()}")
        lines.append(f"  Rows: {dataset_info.get('total_rows', 0):,}")
        lines.append(f"  Completeness: {dataset_info.get('overall_completeness', 0)}%")
        lines.append("")

        # Show columns with nulls
        columns_with_nulls = [
            (col, info)
            for col, info in dataset_info.get("columns", {}).items()
            if info.get("null_count", 0) > 0
        ]

        if columns_with_nulls:
            lines.append("  Columns with missing values:")
            for col, info in columns_with_nulls:
                lines.append(
                    f"    - {col}: {info['null_count']:,} nulls "
                    f"({info['null_percentage']}%) -> {info['strategy']}"
                )
        else:
            lines.append("  No missing values!")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN (for testing)
# =============================================================================
if __name__ == "__main__":
    report = audit_all_datasets()
    print(format_quality_report(report))
