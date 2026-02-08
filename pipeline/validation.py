"""
Validation module for Ask Syracuse Data.
Validates LLM outputs against ground-truth calculations and performs sanity checks.
Supports expanded metrics: count, count_distinct, avg, min, max.
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


def _apply_filters(
    df: pd.DataFrame, filters: Dict[str, Any], metadata: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply intent filters to a dataframe for ground-truth calculation.
    Handles both old format (bare values) and new format ({op, value}).
    """
    if not filters:
        return df

    filtered = df.copy()
    dataset = metadata.get("dataset", "")

    date_cols = {
        "violations": "violation_date",
        "vacant_properties": "completion_date",
        "crime_2022": "dateend",
        "rental_registry": "completion_date",
    }

    for key, filt in filters.items():
        # Normalize: support both old format (bare value) and new format ({op, value})
        if isinstance(filt, dict) and "op" in filt:
            op = filt["op"]
            value = filt["value"]
        else:
            op = "="
            value = filt

        if key == "year":
            date_col = date_cols.get(dataset)
            if date_col and date_col in filtered.columns:
                filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")
                years = filtered[date_col].dt.year
                if op == "=":
                    filtered = filtered[years == value]
                elif op == ">=":
                    filtered = filtered[years >= value]
                elif op == "<=":
                    filtered = filtered[years <= value]
                elif op == ">":
                    filtered = filtered[years > value]
                elif op == "<":
                    filtered = filtered[years < value]
                elif op == "between":
                    filtered = filtered[(years >= value[0]) & (years <= value[1])]
                elif op == "in":
                    filtered = filtered[years.isin(value)]
        elif key in filtered.columns:
            col = filtered[key]
            if op == "=":
                filtered = filtered[col.astype(str).str.lower() == str(value).lower()]
            elif op in (">=", "<=", ">", "<"):
                try:
                    col_num = pd.to_numeric(col, errors="coerce")
                    if op == ">=":
                        filtered = filtered[col_num >= value]
                    elif op == "<=":
                        filtered = filtered[col_num <= value]
                    elif op == ">":
                        filtered = filtered[col_num > value]
                    elif op == "<":
                        filtered = filtered[col_num < value]
                except Exception:
                    pass
            elif op == "between":
                try:
                    col_num = pd.to_numeric(col, errors="coerce")
                    filtered = filtered[(col_num >= value[0]) & (col_num <= value[1])]
                except Exception:
                    pass
            elif op == "in":
                str_vals = [str(v).lower() for v in value]
                filtered = filtered[col.astype(str).str.lower().isin(str_vals)]
            elif op == "like":
                pattern = str(value).replace("%", ".*").replace("_", ".")
                filtered = filtered[col.astype(str).str.match(pattern, case=False, na=False)]

    return filtered


def _resolve_group_by_columns(group_by: Any, df: pd.DataFrame) -> Optional[List[str]]:
    """
    Resolve group_by to actual dataframe column names.
    Temporal groups (year, month, quarter) don't correspond to actual columns,
    so we skip them for ground-truth validation.
    """
    if group_by is None:
        return None

    # Normalize to list
    if isinstance(group_by, str):
        cols = [group_by]
    elif isinstance(group_by, list):
        cols = group_by
    else:
        return None

    # Keep only columns that actually exist in the dataframe
    valid = [c for c in cols if c in df.columns]
    return valid if valid else None


def validate_count_result(
    result_df: pd.DataFrame,
    source_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> ValidationResult:
    """
    Validate count/metric results against ground-truth.
    Supports count, count_distinct, avg, min, max.
    """
    validation = ValidationResult()

    if result_df is None or result_df.empty:
        validation.add_warning("Result is empty - cannot validate")
        return validation

    metric = intent.get("metric", "count")
    group_by = intent.get("group_by")
    filters = intent.get("filters") or {}

    # Apply filters to source dataframe for ground-truth
    filtered_df = _apply_filters(source_df, filters, metadata)

    if filtered_df.empty:
        validation.add_warning("Filtered source data is empty")
        return validation

    # Resolve group_by to actual columns (skip temporal groups)
    group_cols = _resolve_group_by_columns(group_by, filtered_df)

    if metric == "count":
        _validate_count(validation, result_df, filtered_df, group_cols)
    elif metric == "count_distinct":
        distinct_col = intent.get("distinct_column")
        _validate_count_distinct(validation, result_df, filtered_df, group_cols, distinct_col)
    elif metric in ("avg", "min", "max"):
        # Skip ground-truth validation for computed columns (they use DuckDB date_diff)
        validation.add_warning(
            f"Ground-truth validation for '{metric}' on computed columns is approximate"
        )
        validation.ground_truth["total_records"] = len(filtered_df)
    else:
        validation.add_warning(f"Unknown metric '{metric}' - skipping validation")

    return validation


# Alias for backward compatibility
validate_metric_result = validate_count_result


def _validate_count(
    validation: ValidationResult,
    result_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    group_cols: Optional[List[str]],
) -> None:
    """Validate COUNT(*) aggregation."""
    if group_cols:
        ground_truth_counts = filtered_df.groupby(group_cols).size().reset_index(name="count")
        validation.ground_truth["total_groups"] = len(ground_truth_counts)

        # Count non-null records for the first group column
        first_col = group_cols[0]
        validation.ground_truth["total_records"] = int(filtered_df[first_col].notna().sum())

        result_total = result_df["count"].sum() if "count" in result_df.columns else 0
        ground_truth_total = ground_truth_counts["count"].sum()

        if abs(result_total - ground_truth_total) > 0:
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
        ground_truth_count = len(filtered_df)
        validation.ground_truth["total_records"] = ground_truth_count

        if "count" in result_df.columns:
            result_count = result_df["count"].iloc[0] if len(result_df) > 0 else 0
            if result_count != ground_truth_count:
                validation.add_error(
                    f"Total count mismatch: result={result_count:,}, expected={ground_truth_count:,}"
                )


def _validate_count_distinct(
    validation: ValidationResult,
    result_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    group_cols: Optional[List[str]],
    distinct_col: Optional[str],
) -> None:
    """Validate COUNT(DISTINCT ...) aggregation."""
    if not distinct_col or distinct_col not in filtered_df.columns:
        validation.add_warning(f"Cannot validate count_distinct: column '{distinct_col}' not in source")
        return

    if group_cols:
        ground_truth = filtered_df.groupby(group_cols)[distinct_col].nunique().reset_index(
            name="count_distinct"
        )
        validation.ground_truth["total_groups"] = len(ground_truth)

        result_total = (
            result_df["count_distinct"].sum() if "count_distinct" in result_df.columns else 0
        )
        ground_truth_total = ground_truth["count_distinct"].sum()

        if abs(result_total - ground_truth_total) > 0:
            discrepancy_pct = abs(result_total - ground_truth_total) / max(ground_truth_total, 1) * 100
            if discrepancy_pct > 5:
                validation.add_error(
                    f"Count distinct mismatch: result={result_total:,}, expected={ground_truth_total:,} "
                    f"({discrepancy_pct:.1f}% difference)"
                )
    else:
        ground_truth_count = filtered_df[distinct_col].nunique()
        validation.ground_truth["total_distinct"] = ground_truth_count

        if "count_distinct" in result_df.columns:
            result_count = result_df["count_distinct"].iloc[0] if len(result_df) > 0 else 0
            if result_count != ground_truth_count:
                validation.add_error(
                    f"Count distinct mismatch: result={result_count:,}, expected={ground_truth_count:,}"
                )


def validate_join_result(
    result_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    intent: Dict[str, Any],
    join_key_config: Dict[str, str],
) -> ValidationResult:
    """Validate join query results against ground-truth pandas merge."""
    validation = ValidationResult()

    if result_df is None or result_df.empty:
        validation.add_warning("Result is empty - cannot validate")
        return validation

    left_key = join_key_config["left"]
    right_key = join_key_config["right"]

    primary_valid = primary_df[left_key].notna().sum()
    secondary_valid = secondary_df[right_key].notna().sum()

    validation.ground_truth["primary_records_with_key"] = int(primary_valid)
    validation.ground_truth["secondary_records_with_key"] = int(secondary_valid)

    merged = pd.merge(
        primary_df[[left_key]].dropna(),
        secondary_df[[right_key]].dropna(),
        left_on=left_key,
        right_on=right_key,
        how="inner",
    )
    validation.ground_truth["matched_records"] = len(merged)

    if "count" in result_df.columns:
        result_total = result_df["count"].sum()
        if result_total > len(merged) * 1.1:
            validation.add_warning(
                f"Join result count ({result_total:,}) exceeds expected matches ({len(merged):,})"
            )

    return validation


def sanity_check_result(
    result_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> ValidationResult:
    """Perform general sanity checks on query results."""
    validation = ValidationResult()

    if result_df is None or result_df.empty:
        return validation

    # Determine the metric column name to check
    metric = intent.get("metric", "count")
    if metric == "count_distinct":
        count_col = "count_distinct"
    elif metric in ("avg", "min", "max"):
        metric_column = intent.get("metric_column", "")
        count_col = f"{metric}_{metric_column}"
    else:
        count_col = "count"

    # Check for suspiciously high counts (only for count metrics)
    if count_col in result_df.columns and metric in ("count", "count_distinct"):
        max_count = result_df[count_col].max()
        total_count = result_df[count_col].sum()

        if total_count > 500000:
            validation.add_warning(
                f"Unusually high total count ({total_count:,}) - verify query logic"
            )

        if len(result_df) > 1:
            mean_count = result_df[count_col].mean()
            if max_count > mean_count * 10 and mean_count > 100:
                validation.add_warning(
                    f"Possible outlier: max count ({max_count:,}) is 10x+ the average ({mean_count:,.0f})"
                )

    # Check for unexpected null groups
    group_by = intent.get("group_by")
    if group_by:
        # group_by can be a list or string
        cols_to_check = group_by if isinstance(group_by, list) else [group_by]
        for col in cols_to_check:
            if col in result_df.columns:
                null_groups = result_df[col].isna().sum()
                if null_groups > 0:
                    validation.add_warning(
                        f"{null_groups} record(s) have null/missing {col} values"
                    )

    return validation


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
    "validate_metric_result",
    "validate_join_result",
    "sanity_check_result",
    "combine_validations",
]
