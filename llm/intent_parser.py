"""
Intent parsing layer: converts natural language questions into structured intents.
LLM is used only for parsing; no data access or analysis occurs here.
Supports expanded metrics, temporal grouping, distinct counts, having, and advanced SQL routing.
"""
from __future__ import annotations
import json
import re
from typing import Any, Callable, Dict, Optional

from .prompt_templates import NL_TO_INTENT_PROMPT, NL_TO_JOIN_INTENT_PROMPT

LLMCallable = Callable[[str], str]


class IntentParsingError(Exception):
    """Raised when an intent cannot be parsed or validated."""


def _is_join_query(question: str) -> bool:
    """Detect if a question requires a cross-dataset join."""
    q = question.lower()

    # Exclude patterns that look like joins but are actually single-dataset queries
    # "unique/distinct properties" is a count_distinct, not a join
    if any(w in q for w in ["unique", "distinct", "different"]):
        return False
    # "neighborhoods with more than N violations" is a HAVING filter
    if re.search(r"with (?:more than|over|at least|fewer than|less than|under) \d+", q):
        return False
    # "Part 1 vs Part 2" is a within-dataset comparison, not a cross-dataset join
    if "part" in q and "vs" in q:
        return False

    join_patterns = [
        # rental + violations
        ("rental" in q and "violation" in q),
        ("rental" in q and "code" in q and "violation" in q),
        # vacant + violations
        ("vacant" in q and "violation" in q and "unfit" not in q),
        # rental + vacant
        ("rental" in q and "vacant" in q),
        # crime + violations
        ("crime" in q and "violation" in q),
        # crime + vacant
        ("crime" in q and "vacant" in q),
        # "vs" comparisons involving crime
        ("crime" in q and "vs" in q),
        # unfit + violations
        ("unfit" in q and "violation" in q),
        # unfit + vacant
        ("unfit" in q and "vacant" in q),
        # Generic: "properties with violations" when rental/vacant context
        ("propert" in q and "violation" in q and ("rental" in q or "vacant" in q or "registered" in q)),
        # "with" connecting two different dataset keywords
        ("with" in q and "rental" in q and "violation" in q),
        ("with" in q and "vacant" in q and "violation" in q),
        ("with" in q and "rental" in q and "vacant" in q),
    ]

    return any(join_patterns)


def _needs_advanced_sql(question: str) -> bool:
    """Detect if a question requires LLM-generated SQL (too complex for intent pipeline)."""
    q = question.lower()

    advanced_patterns = [
        "rank" in q and ("by" in q or "top" in q),
        "percentile" in q,
        "top" in q and "percent" in q,
        "running total" in q,
        "cumulative" in q,
        "compared to average" in q or "above average" in q or "below average" in q,
        "rate per" in q or "per capita" in q,
        "ratio" in q,
        "percent change" in q or "percentage change" in q,
        "rolling" in q and ("average" in q or "mean" in q),
        "moving average" in q,
        "year over year" in q or "yoy" in q,
        "standard deviation" in q or "std dev" in q,
        "median" in q,
        "correlation" in q,
    ]

    return any(advanced_patterns)


# =============================================================================
# HEURISTIC HELPERS
# =============================================================================

def _extract_threshold(question: str) -> Optional[Dict[str, Any]]:
    """Extract having-clause threshold from question text.
    E.g., "more than 100" -> {"op": ">", "value": 100}
    """
    q = question.lower()

    patterns = [
        (r"more than (\d+)", ">"),
        (r"over (\d+)", ">"),
        (r"at least (\d+)", ">="),
        (r"greater than (\d+)", ">"),
        (r"fewer than (\d+)", "<"),
        (r"less than (\d+)", "<"),
        (r"under (\d+)", "<"),
        (r"at most (\d+)", "<="),
        (r"> ?(\d+)", ">"),
        (r">= ?(\d+)", ">="),
        (r"< ?(\d+)", "<"),
        (r"<= ?(\d+)", "<="),
    ]

    for pattern, op in patterns:
        match = re.search(pattern, q)
        if match:
            return {"op": op, "value": int(match.group(1))}

    return None


def _extract_year_filter(question: str) -> Optional[Dict[str, Any]]:
    """Extract year-based filter from question text.
    E.g., "since 2020" -> {"op": ">=", "value": 2020}
    """
    q = question.lower()

    # "between YYYY and YYYY"
    match = re.search(r"between (\d{4}) and (\d{4})", q)
    if match:
        return {"op": "between", "value": [int(match.group(1)), int(match.group(2))]}

    # "since YYYY" or "after YYYY"
    match = re.search(r"(?:since|after|from) (\d{4})", q)
    if match:
        return {"op": ">=", "value": int(match.group(1))}

    # "before YYYY" or "until YYYY"
    match = re.search(r"(?:before|until|up to) (\d{4})", q)
    if match:
        return {"op": "<=", "value": int(match.group(1))}

    # "in YYYY" (specific year)
    match = re.search(r"in (\d{4})\b", q)
    if match:
        return {"op": "=", "value": int(match.group(1))}

    return None


def _detect_temporal_group(question: str) -> Optional[str]:
    """Detect temporal grouping keywords."""
    q = question.lower()

    if any(w in q for w in ["by year", "per year", "yearly", "annual", "each year", "over time", "over the years"]):
        return "year"
    if any(w in q for w in ["by month", "per month", "monthly", "each month"]):
        return "month"
    if any(w in q for w in ["by quarter", "quarterly", "per quarter"]):
        return "quarter"
    if "trend" in q or "over time" in q:
        return "year"  # default temporal for "trend" queries

    return None


def _detect_metric(question: str) -> tuple:
    """Detect the metric type and related column from the question.
    Returns (metric, metric_column, distinct_column).
    """
    q = question.lower()

    # "most common X" means count + group_by, NOT count_distinct
    if "most common" in q:
        return "count", None, None

    # Count distinct
    if any(w in q for w in ["unique", "distinct", "different"]):
        # Determine which column to count distinct
        if "propert" in q or "address" in q or "sbl" in q:
            distinct_col = "sbl"
        elif "neighborhood" in q:
            distinct_col = "neighborhood"
        elif "zip" in q:
            # Pick the right zip col based on context
            if "violation" in q:
                distinct_col = "complaint_zip"
            else:
                distinct_col = "zip"
        else:
            distinct_col = "sbl"  # default
        return "count_distinct", None, distinct_col

    # Average on computed columns
    if any(w in q for w in ["average", "avg", "mean"]):
        if any(w in q for w in ["comply", "compliance", "days to comply"]):
            return "avg", "days_to_comply", None
        if any(w in q for w in ["open", "days open", "how long"]):
            return "avg", "days_open", None
        if any(w in q for w in ["duration", "cert", "certificate"]):
            return "avg", "cert_duration_days", None

    # Min/max
    if any(w in q for w in ["fastest", "shortest", "minimum", "min"]):
        if "comply" in q or "compliance" in q:
            return "min", "days_to_comply", None
        if "open" in q:
            return "min", "days_open", None
    if any(w in q for w in ["slowest", "longest", "maximum", "max"]):
        if "comply" in q or "compliance" in q:
            return "max", "days_to_comply", None
        if "open" in q:
            return "max", "days_open", None

    return "count", None, None


# =============================================================================
# JOIN INTENT HEURISTICS
# =============================================================================

def _heuristic_join_intent(question: str) -> Optional[Dict[str, Any]]:
    """Deterministic fallback for common join queries when LLM is unavailable."""
    q = question.lower()

    # vacant + violations (check first since "properties" could match rental)
    if "vacant" in q and "violation" in q:
        join_type = "sbl" if ("specific" in q or "how many" in q or "count" in q) else "zip"
        group_by = "zip" if join_type == "zip" else None
        return {
            "query_type": "join",
            "primary_dataset": "vacant_properties",
            "secondary_dataset": "violations",
            "join_type": join_type,
            "metric": "count",
            "group_by": group_by,
            "filters": {},
            "limit": None,
        }

    # rental + violations
    if ("rental" in q or "propert" in q) and "violation" in q:
        join_type = "sbl" if ("specific" in q or "property" in q or "address" in q) else "zip"
        group_by = "zip" if join_type == "zip" else None
        return {
            "query_type": "join",
            "primary_dataset": "rental_registry",
            "secondary_dataset": "violations",
            "join_type": join_type,
            "metric": "count",
            "group_by": group_by,
            "filters": {},
            "limit": 20 if join_type == "sbl" else None,
        }

    # rental + vacant
    if "rental" in q and "vacant" in q:
        return {
            "query_type": "join",
            "primary_dataset": "rental_registry",
            "secondary_dataset": "vacant_properties",
            "join_type": "sbl",
            "metric": "count",
            "group_by": None,
            "filters": {},
            "limit": 20,
        }

    # crime + violations
    if "crime" in q and "violation" in q:
        join_type = "zip" if "zip" in q else "neighborhood"
        return {
            "query_type": "join",
            "primary_dataset": "crime",
            "secondary_dataset": "violations",
            "join_type": join_type,
            "metric": "count",
            "group_by": None,
            "filters": {},
            "limit": None,
        }

    # crime + vacant
    if "crime" in q and "vacant" in q:
        join_type = "zip" if "zip" in q else "neighborhood"
        return {
            "query_type": "join",
            "primary_dataset": "crime",
            "secondary_dataset": "vacant_properties",
            "join_type": join_type,
            "metric": "count",
            "group_by": None,
            "filters": {},
            "limit": None,
        }

    # unfit + violations
    if "unfit" in q and "violation" in q:
        join_type = "sbl" if ("specific" in q or "how many" in q or "count" in q) else "zip"
        group_by = "zip" if join_type == "zip" else None
        return {
            "query_type": "join",
            "primary_dataset": "unfit_properties",
            "secondary_dataset": "violations",
            "join_type": join_type,
            "metric": "count",
            "group_by": group_by,
            "filters": {},
            "limit": None,
        }

    # unfit + vacant
    if "unfit" in q and "vacant" in q:
        return {
            "query_type": "join",
            "primary_dataset": "unfit_properties",
            "secondary_dataset": "vacant_properties",
            "join_type": "sbl",
            "metric": "count",
            "group_by": None,
            "filters": {},
            "limit": 20,
        }

    # crime + "vs" (generic comparison)
    if "crime" in q and "vs" in q:
        if "vacant" in q:
            secondary = "vacant_properties"
        elif "violation" in q:
            secondary = "violations"
        else:
            secondary = "vacant_properties"
        join_type = "zip" if "zip" in q else "neighborhood"
        return {
            "query_type": "join",
            "primary_dataset": "crime",
            "secondary_dataset": secondary,
            "join_type": join_type,
            "metric": "count",
            "group_by": None,
            "filters": {},
            "limit": None,
        }

    return None


# =============================================================================
# SINGLE-DATASET INTENT HEURISTICS
# =============================================================================

def _heuristic_intent(question: str) -> Optional[Dict[str, Any]]:
    """Deterministic fallback for common single-dataset and join questions."""
    q = question.lower()

    # Check for join queries first
    if _is_join_query(q):
        return _heuristic_join_intent(question)

    # Detect metric type
    metric, metric_column, distinct_column = _detect_metric(question)

    # Detect temporal grouping
    temporal_group = _detect_temporal_group(question)

    # Detect having clause
    having = _extract_threshold(question)

    # Detect year filter
    year_filter = _extract_year_filter(question)

    # Single-dataset queries
    if "violation" in q:
        group_by = None
        if "neighborhood" in q:
            group_by = "neighborhood"
        elif "zip" in q:
            group_by = "complaint_zip"
        elif "status" in q:
            group_by = "status_type_name"
        elif "type" in q:
            group_by = "violation"

        # Add temporal group if detected
        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        intent = {
            "dataset": "violations",
            "metric": metric,
            "group_by": group_by,
            "filters": filters,
        }
        if metric_column:
            intent["metric_column"] = metric_column
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    if "vacant" in q:
        group_by = None
        if "neighborhood" in q:
            group_by = "neighborhood"
        elif "zip" in q:
            group_by = "zip"

        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        intent = {
            "dataset": "vacant_properties",
            "metric": metric,
            "group_by": group_by,
            "filters": filters,
        }
        if metric_column:
            intent["metric_column"] = metric_column
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    if "rental" in q:
        group_by = None
        if "zip" in q or "neighborhood" in q:
            group_by = "zip"  # rental_registry has no neighborhood; use ZIP as best proxy

        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        intent = {
            "dataset": "rental_registry",
            "metric": metric,
            "group_by": group_by,
            "filters": filters,
        }
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    if ("crime" in q or "offense" in q or "arrest" in q) and "parking" not in q:
        group_by = None
        if "part 1" in q and "part 2" in q:
            group_by = "crime_part"
        elif "neighborhood" in q:
            group_by = "neighborhood"
        elif "zip" in q:
            group_by = "zip"
        elif "type" in q or "code" in q or "kind" in q:
            group_by = "code_defined"

        # "how many arrests" / "arrests made" -> filter arrest=Yes, don't group by arrest
        arrest_as_filter = ("arrest" in q and
                            any(w in q for w in ["how many arrest", "arrests made",
                                                  "arrests in", "total arrest",
                                                  "number of arrest"]))
        if not arrest_as_filter and "arrest" in q and group_by is None:
            group_by = "arrest"

        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group
        elif group_by is None:
            group_by = "code_defined"  # default to crime type when no other grouping

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        # Arrest filtering: "arrests by neighborhood", "how many arrests in 2024"
        if arrest_as_filter:
            filters["arrest"] = {"op": "=", "value": "Yes"}
        elif "arrest" in q and group_by != "arrest":
            filters["arrest"] = {"op": "=", "value": "Yes"}
        elif "arrest" in q and group_by == "arrest" and "neighborhood" in q:
            group_by = ["neighborhood", "arrest"]

        intent = {
            "dataset": "crime",
            "metric": metric,
            "group_by": group_by,
            "filters": filters,
        }
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    # --- Unfit Properties ---
    if "unfit" in q:
        group_by = None
        if "zip" in q:
            group_by = "zip"
        elif "status" in q:
            group_by = "status_type_name"
        elif "vacant" in q:
            group_by = "vacant"
        elif "type" in q:
            group_by = "violation"

        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        intent = {"dataset": "unfit_properties", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    # --- Trash Pickup ---
    if "trash" in q or "garbage" in q or "sanitation" in q or "recycling" in q:
        group_by = None
        if "zip" in q:
            group_by = "zip"
        elif "day" in q or "schedule" in q or "monday" in q or "tuesday" in q or "wednesday" in q or "thursday" in q or "friday" in q:
            group_by = "sanitation"
        elif "recycl" in q or "week" in q:
            group_by = "recyclingw"

        # Handle day-specific filter
        filters = {}
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            if day in q:
                filters["sanitation"] = {"op": "=", "value": day.capitalize()}

        intent = {"dataset": "trash_pickup", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        return intent

    # --- Historical Properties ---
    if "historical" in q or "landmark" in q or "national register" in q:
        group_by = None
        if "zip" in q:
            group_by = "zip"
        elif "eligible" in q or "national register" in q or "nr" in q:
            group_by = "nr_eligible"
        elif "landmark" in q or "lpss" in q:
            group_by = "lpss"

        filters = {}
        if "eligible" in q and group_by != "nr_eligible":
            filters["nr_eligible"] = {"op": "in", "value": ["NR Listed", "NR Eligible (SHPO)"]}

        intent = {"dataset": "historical_properties", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        return intent

    # --- Assessment Roll ---
    if "assessment" in q or ("property" in q and ("value" in q or "class" in q)):
        group_by = None
        if "class" in q or "type" in q:
            group_by = "prop_class_description"

        # Detect avg/min/max on total_assessment
        if any(w in q for w in ["average", "avg", "mean"]):
            metric, metric_column = "avg", "total_assessment"
        elif any(w in q for w in ["minimum", "min", "lowest"]):
            metric, metric_column = "min", "total_assessment"
        elif any(w in q for w in ["maximum", "max", "highest"]):
            metric, metric_column = "max", "total_assessment"
        else:
            metric_column = None

        filters = {}
        if "residential" in q:
            filters["prop_class_description"] = {"op": "like", "value": "%Residential%"}
        elif "commercial" in q:
            filters["prop_class_description"] = {"op": "like", "value": "%Commercial%"}

        intent = {"dataset": "assessment_roll", "metric": metric, "group_by": group_by, "filters": filters}
        if metric_column:
            intent["metric_column"] = metric_column
        if distinct_column:
            intent["distinct_column"] = distinct_column
        return intent

    # --- SYRCityline / Service Requests ---
    if "cityline" in q or "service request" in q or "311" in q or "complaint" in q:
        group_by = "category"  # default grouping
        if "zip" in q:
            group_by = "zip"
        elif "agency" in q:
            group_by = "agency_name"

        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        intent = {"dataset": "cityline_requests", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    # --- Snow Routes ---
    if "snow" in q and "route" in q:
        group_by = "zip" if "zip" in q else None

        filters = {}
        intent = {"dataset": "snow_routes", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        return intent

    # --- Bike Suitability ---
    if "bike" in q and ("suitab" in q or "rating" in q):
        group_by = "bike_suitability_19"
        intent = {"dataset": "bike_suitability", "metric": metric, "group_by": group_by, "filters": {}}
        return intent

    # --- Bike Infrastructure ---
    if "bike" in q and ("infra" in q or "lane" in q or "trail" in q or "mile" in q or "path" in q):
        group_by = "infrastructure_type"
        # "how many miles" -> sum metric
        if "mile" in q or "length" in q or "total" in q:
            metric = "sum"
            intent = {"dataset": "bike_infrastructure", "metric": metric, "metric_column": "length_mi", "group_by": group_by, "filters": {}}
        else:
            intent = {"dataset": "bike_infrastructure", "metric": metric, "group_by": group_by, "filters": {}}
        return intent

    # --- Parking Violations ---
    if "parking" in q:
        group_by = "description"  # default to violation type
        if "zip" in q:
            group_by = "zip"
        elif "status" in q:
            group_by = "status"

        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        intent = {"dataset": "parking_violations", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    # --- Permit Requests ---
    if "permit" in q or "building permit" in q:
        group_by = "permit_type"
        if "zip" in q:
            group_by = "zip"

        if temporal_group:
            if group_by:
                group_by = [group_by, temporal_group]
            else:
                group_by = temporal_group

        filters = {}
        if year_filter:
            filters["year"] = year_filter

        intent = {"dataset": "permit_requests", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    # --- Tree Inventory ---
    if "tree" in q:
        group_by = "spp_com"  # default to species
        if "neighborhood" in q or "area" in q:
            group_by = "area"
        elif "zip" in q:
            group_by = "zip"

        filters = {}
        intent = {"dataset": "tree_inventory", "metric": metric, "group_by": group_by, "filters": filters}
        if distinct_column:
            intent["distinct_column"] = distinct_column
        if having:
            intent["having"] = having
        return intent

    # --- Lead Testing ---
    if "lead" in q and ("test" in q or "poison" in q):
        group_by = "census_tract"
        intent = {"dataset": "lead_testing", "metric": metric, "group_by": group_by, "filters": {}}
        return intent

    return None


# =============================================================================
# MAIN PARSER
# =============================================================================

def parse_intent(question: str, llm: Optional[LLMCallable] = None) -> Dict[str, Any]:
    """
    Convert a natural language question into a structured intent dict.
    If an LLM callable is provided, it must return a JSON string.
    A deterministic heuristic is used as a fallback when no LLM is supplied.

    Returns either:
    - A standard intent dict (for single or join queries)
    - {"query_path": "advanced_sql", "question": question} for complex queries needing LLM SQL
    """
    if not question or not question.strip():
        raise IntentParsingError("Question is empty or missing.")

    # Check if this needs the advanced SQL path (only when LLM is available)
    if llm and _needs_advanced_sql(question):
        return {"query_path": "advanced_sql", "question": question}

    intent: Optional[Dict[str, Any]] = None

    if llm:
        # Detect if this is a join query
        is_join = _is_join_query(question)
        prompt_template = NL_TO_JOIN_INTENT_PROMPT if is_join else NL_TO_INTENT_PROMPT
        prompt = prompt_template + f"\nQuestion: {question}\nJSON:"
        raw = llm(prompt)
        try:
            intent = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IntentParsingError(f"LLM did not return valid JSON: {exc}") from exc
    else:
        intent = _heuristic_intent(question)
        if intent is None:
            # Provide a helpful suggestion instead of a generic error
            suggestions = []
            q = question.lower()
            if any(w in q for w in ["safest", "best", "worst", "should i", "recommend"]):
                suggestions.append(
                    "This looks like a subjective or recommendation question. "
                    "Try asking about specific data instead, e.g. "
                    "'Crime by neighborhood' or 'Violations by zip code'."
                )
            raise IntentParsingError(
                "Unable to understand this question. "
                + (suggestions[0] if suggestions else
                   "Try rephrasing with a specific dataset like 'violations', 'crime', 'parking', 'trees', etc.")
            )

    return intent


__all__ = ["parse_intent", "IntentParsingError", "_is_join_query", "_needs_advanced_sql"]
