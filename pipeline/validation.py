"""
Validation module for Ask Syracuse Data.
Validates LLM outputs against ground-truth calculations and performs sanity checks.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.passed = True
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.ground_truth: Dict[str, Any] = {}

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "ground_truth": self.ground_truth,
        }


def validate_count_result(
    result_df: pd.DataFrame,
    source_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> ValidationResult:
    """
    Validate count aggregation results against ground-truth.
    Compares LLM-generated query results to direct pandas calculations.
    """
    validation = ValidationResult()

    if result_df is None or result_df.empty:
        validation.add_warning("Result is empty - cannot validate")
        return validation

    group_by = intent.get("group_by")
    filters = intent.get("filters") or {}

    # Apply filters to source dataframe for ground-truth calculation
    filtered_df = _apply_filters(source_df, filters, metadata)

    if filtered_df.empty:
        validation.add_warning("Filtered source data is empty")
        return validation

    # Ground-truth calculation
    if group_by:
        # Grouped count
        ground_truth_counts = filtered_df.groupby(group_by).size().reset_index(name='count')
        validation.ground_truth["total_groups"] = len(ground_truth_counts)
        validation.ground_truth["total_records"] = int(filtered_df[group_by].notna().sum())

        # Compare total count
        result_total = result_df["count"].sum() if "count" in result_df.columns else 0
        ground_truth_total = ground_truth_counts["count"].sum()

        if abs(result_total - ground_truth_total) > 0:
            # Allow small discrepancies due to null handling
            discrepancy_pct = abs(result_total - ground_truth_total) / max(ground_truth_total, 1) * 100
            if discrepancy_pct > 5:
                validation.add_error(
                    f"Count mismatch: result={result_total:,}, expected={ground_truth_total:,} "
                    f"({discrepancy_pct:.1f}% difference)"
                )
            elif discrepancy_pct > 0:
                validation.add_warning(
                    f"Minor count discrepancy ({discrepancy_pct:.1f}%) - likely due to null handling"
                )
    else:
        # Total count
        ground_truth_count = len(filtered_df)
        validation.ground_truth["total_records"] = ground_truth_count

        if "count" in result_df.columns:
            result_count = result_df["count"].iloc[0] if len(result_df) > 0 else 0
            if result_count != ground_truth_count:
                validation.add_error(
                    f"Total count mismatch: result={result_count:,}, expected={ground_truth_count:,}"
                )

    return validation


def validate_join_result(
    result_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    intent: Dict[str, Any],
    join_key_config: Dict[str, str],
) -> ValidationResult:
    """
    Validate join query results against ground-truth pandas merge.
    """
    validation = ValidationResult()

    if result_df is None or result_df.empty:
        validation.add_warning("Result is empty - cannot validate")
        return validation

    join_type = intent.get("join_type", "zip")
    left_key = join_key_config["left"]
    right_key = join_key_config["right"]

    # Ground-truth: count matching records
    primary_valid = primary_df[left_key].notna().sum()
    secondary_valid = secondary_df[right_key].notna().sum()

    validation.ground_truth["primary_records_with_key"] = int(primary_valid)
    validation.ground_truth["secondary_records_with_key"] = int(secondary_valid)

    # Perform ground-truth merge
    merged = pd.merge(
        primary_df[[left_key]].dropna(),
        secondary_df[[right_key]].dropna(),
        left_on=left_key,
        right_on=right_key,
        how="inner"
    )
    validation.ground_truth["matched_records"] = len(merged)

    # Sanity check: result should not exceed ground-truth match count
    if "count" in result_df.columns:
        result_total = result_df["count"].sum()
        if result_total > len(merged) * 1.1:  # Allow 10% margin for aggregation differences
            validation.add_warning(
                f"Join result count ({result_total:,}) exceeds expected matches ({len(merged):,})"
            )

    return validation


def sanity_check_result(
    result_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> ValidationResult:
    """
    Perform general sanity checks on query results.
    """
    validation = ValidationResult()

    if result_df is None or result_df.empty:
        return validation

    # Check for suspiciously high counts
    if "count" in result_df.columns:
        max_count = result_df["count"].max()
        total_count = result_df["count"].sum()

        # Syracuse city population ~150k, most datasets < 200k records
        if total_count > 500000:
            validation.add_warning(
                f"Unusually high total count ({total_count:,}) - verify query logic"
            )

        # Check for extreme outliers within grouped data
        if len(result_df) > 1:
            mean_count = result_df["count"].mean()
            if max_count > mean_count * 10 and mean_count > 100:
                validation.add_warning(
                    f"Possible outlier: max count ({max_count:,}) is 10x+ the average ({mean_count:,.0f})"
                )

    # Check for unexpected null groups
    group_by = intent.get("group_by")
    if group_by and group_by in result_df.columns:
        null_groups = result_df[group_by].isna().sum()
        if null_groups > 0:
            validation.add_warning(
                f"{null_groups} record(s) have null/missing {group_by} values"
            )

    # Check row count vs limit
    limit = intent.get("limit")
    if limit and len(result_df) < limit:
        # This is fine - just informational
        pass

    return validation


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any], metadata: Dict[str, Any]) -> pd.DataFrame:
    """Apply intent filters to a dataframe for ground-truth calculation."""
    if not filters:
        return df

    filtered = df.copy()
    dataset = metadata.get("dataset", "")

    for key, value in filters.items():
        if key == "year":
            # Find date column based on dataset
            date_cols = {
                "violations": "violation_date",
                "vacant_properties": "completion_date",
                "crime_2022": "dateend",
                "rental_registry": "completion_date",
            }
            date_col = date_cols.get(dataset)
            if date_col and date_col in filtered.columns:
                filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")
                filtered = filtered[filtered[date_col].dt.year == value]
        elif key in filtered.columns:
            filtered = filtered[filtered[key].astype(str).str.lower() == str(value).lower()]

    return filtered


def combine_validations(*validations: ValidationResult) -> ValidationResult:
    """Combine multiple validation results into one."""
    combined = ValidationResult()

    for v in validations:
        combined.warnings.extend(v.warnings)
        combined.errors.extend(v.errors)
        combined.ground_truth.update(v.ground_truth)
        if not v.passed:
            combined.passed = False

    return combined


__all__ = [
    "ValidationResult",
    "validate_count_result",
    "validate_join_result",
    "sanity_check_result",
    "combine_validations",
]
