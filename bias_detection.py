"""
Bias detection module for Ask Syracuse Data.
Identifies potential biases in query framing, data interpretation, and result presentation.
"""
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd


class BiasWarning:
    """Container for a single bias warning."""

    def __init__(self, bias_type: str, message: str, severity: str = "info"):
        self.bias_type = bias_type
        self.message = message
        self.severity = severity  # "info", "warning", "caution"

    def to_dict(self) -> Dict[str, str]:
        return {
            "type": self.bias_type,
            "message": self.message,
            "severity": self.severity,
        }


class BiasDetectionResult:
    """Container for bias detection results."""

    def __init__(self):
        self.warnings: List[BiasWarning] = []

    def add(self, bias_type: str, message: str, severity: str = "info"):
        self.warnings.append(BiasWarning(bias_type, message, severity))

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_list(self) -> List[Dict[str, str]]:
        return [w.to_dict() for w in self.warnings]

    def to_text(self) -> str:
        """Return formatted text for display."""
        if not self.warnings:
            return ""
        lines = []
        for w in self.warnings:
            icon = {"info": "ℹ️", "warning": "⚠️", "caution": "🔍"}.get(w.severity, "•")
            lines.append(f"{icon} {w.message}")
        return "\n".join(lines)


def detect_framing_bias(question: str, intent: Dict[str, Any]) -> BiasDetectionResult:
    """
    Detect potential framing bias in how the question is phrased.
    """
    result = BiasDetectionResult()
    q = question.lower()

    # Leading language detection
    leading_words = ["most", "worst", "best", "highest", "lowest", "dangerous", "safest"]
    for word in leading_words:
        if word in q:
            result.add(
                "framing",
                f"Query uses potentially leading term '{word}' - results are descriptive counts only, not rankings of quality or safety.",
                "caution"
            )
            break

    # Causal language detection
    causal_words = ["cause", "because", "due to", "result of", "leads to", "affects"]
    for word in causal_words:
        if word in q:
            result.add(
                "framing",
                "This data shows correlations, not causation. Administrative records reflect reporting patterns, not underlying causes.",
                "warning"
            )
            break

    # Absolute language detection
    if "all" in q.split() or "every" in q:
        result.add(
            "framing",
            "Data may not capture 'all' instances - records reflect reported/documented cases only.",
            "info"
        )

    return result


def detect_normalization_bias(
    result_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> BiasDetectionResult:
    """
    Detect when raw counts might be misleading without normalization.
    """
    result = BiasDetectionResult()

    group_by = intent.get("group_by")

    # Neighborhood comparisons need population context
    if group_by in ["neighborhood", "zip", "complaint_zip"]:
        result.add(
            "normalization",
            f"Raw counts by {group_by} may not account for population differences. Higher counts may reflect larger populations, not higher rates.",
            "warning"
        )

    # Time-based comparisons
    filters = intent.get("filters") or {}
    if "year" in filters:
        result.add(
            "normalization",
            "Single-year data may not represent typical patterns. Consider multi-year trends for more robust conclusions.",
            "info"
        )

    # Cross-dataset joins
    if metadata.get("query_type") == "join":
        result.add(
            "normalization",
            "Cross-dataset joins show overlap, not rates. A neighborhood with more rental properties will naturally have more rental-violation matches.",
            "warning"
        )

    return result


def detect_selection_bias(
    result_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> BiasDetectionResult:
    """
    Detect potential selection/sampling bias in the data.
    """
    result = BiasDetectionResult()
    dataset = metadata.get("dataset", "")

    # Dataset-specific selection bias warnings
    dataset_biases = {
        "violations": (
            "Code violations reflect enforcement patterns. Areas with more inspections "
            "may show more violations regardless of actual conditions."
        ),
        "crime_2022": (
            "Crime data reflects reported incidents. Under-reporting varies by crime type "
            "and neighborhood, affecting apparent patterns."
        ),
        "vacant_properties": (
            "Vacancy records are administratively identified. Actual vacancy rates may differ "
            "from official designations."
        ),
        "rental_registry": (
            "Rental registry data covers registered properties. Unregistered rentals are not included, "
            "which may vary by neighborhood."
        ),
    }

    if dataset in dataset_biases:
        result.add("selection", dataset_biases[dataset], "info")

    return result


def detect_missing_context(
    result_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> BiasDetectionResult:
    """
    Identify important context that may be missing from the analysis.
    """
    result = BiasDetectionResult()

    row_count = metadata.get("row_count", len(result_df) if result_df is not None else 0)
    group_by = intent.get("group_by")

    # Small sample warning
    if row_count < 5 and group_by:
        result.add(
            "context",
            f"Only {row_count} groups returned. Small samples may not be representative.",
            "warning"
        )

    # No grouping on count query
    if not group_by and metadata.get("query_type") != "join":
        result.add(
            "context",
            "Total count provides limited insight. Consider grouping by neighborhood or time period for more actionable analysis.",
            "info"
        )

    # Limited time range for crime data
    dataset = metadata.get("dataset", "")
    if dataset == "crime_2022":
        result.add(
            "context",
            "Crime data covers 2022 only. Single-year data may not reflect long-term trends.",
            "info"
        )

    return result


def detect_uncertainty(
    result_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> BiasDetectionResult:
    """
    Identify sources of uncertainty in the results.
    """
    result = BiasDetectionResult()

    if result_df is None or result_df.empty:
        return result

    # Check for null/unknown values in results
    group_by = intent.get("group_by")
    if group_by and group_by in result_df.columns:
        null_count = result_df[group_by].isna().sum()
        unknown_count = (result_df[group_by].astype(str).str.lower() == "unknown").sum()
        total_unknown = null_count + unknown_count

        if total_unknown > 0:
            result.add(
                "uncertainty",
                f"{total_unknown} records have unknown/missing {group_by} values and are grouped separately.",
                "info"
            )

    # Large result sets have sampling considerations
    if "count" in result_df.columns:
        total = result_df["count"].sum()
        if total > 10000:
            result.add(
                "uncertainty",
                f"Results aggregate {total:,} records. Individual case details are not visible at this level.",
                "info"
            )

    return result


def run_all_bias_checks(
    question: str,
    result_df: pd.DataFrame,
    intent: Dict[str, Any],
    metadata: Dict[str, Any],
) -> BiasDetectionResult:
    """
    Run all bias detection checks and combine results.
    """
    combined = BiasDetectionResult()

    checks = [
        detect_framing_bias(question, intent),
        detect_normalization_bias(result_df, intent, metadata),
        detect_selection_bias(result_df, intent, metadata),
        detect_missing_context(result_df, intent, metadata),
        detect_uncertainty(result_df, intent, metadata),
    ]

    for check in checks:
        combined.warnings.extend(check.warnings)

    return combined


__all__ = [
    "BiasWarning",
    "BiasDetectionResult",
    "detect_framing_bias",
    "detect_normalization_bias",
    "detect_selection_bias",
    "detect_missing_context",
    "detect_uncertainty",
    "run_all_bias_checks",
]
