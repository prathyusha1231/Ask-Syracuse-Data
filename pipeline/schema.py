"""
Schema and guardrails for Ask Syracuse Data.
Defines allowed datasets, fields, and validates intents.
Supports expanded metrics, temporal grouping, computed columns, and richer filters.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Union

DATASETS: Dict[str, Dict[str, Any]] = {
    "violations": {
        "table": "violations",
        "date_column": "violation_date",
        "allowed_metrics": ["count", "count_distinct", "avg", "min", "max"],
        "allowed_group_by": [
            "neighborhood", "complaint_zip", "status_type_name", "violation",
            "year", "month", "quarter",
        ],
        "allowed_filters": {
            "year": "int", "neighborhood": "text",
            "complaint_zip": "text", "status_type_name": "text",
        },
        "allowed_distinct_columns": ["sbl", "complaint_address", "neighborhood", "complaint_zip"],
        "computed_columns": {
            "days_to_comply": {
                "expr": "date_diff('day', violation_date, comply_by_date)",
                "type": "numeric",
                "null_filter": "violation_date IS NOT NULL AND comply_by_date IS NOT NULL",
            },
            "days_open": {
                "expr": "date_diff('day', open_date, status_date)",
                "type": "numeric",
                "null_filter": "open_date IS NOT NULL AND status_date IS NOT NULL",
            },
        },
        "temporal_group_map": {
            "year": ("violation_date", "year"),
            "month": ("violation_date", "month"),
            "quarter": ("violation_date", "quarter"),
        },
    },
    "vacant_properties": {
        "table": "vacant_properties",
        "date_column": "completion_date",
        "allowed_metrics": ["count", "count_distinct", "avg", "min", "max"],
        "allowed_group_by": [
            "neighborhood", "zip", "vpr_valid", "vpr_result",
            "year", "month", "quarter",
        ],
        "allowed_filters": {
            "year": "int", "neighborhood": "text",
            "zip": "text", "vpr_valid": "text",
        },
        "allowed_distinct_columns": ["sbl", "propertyaddress", "neighborhood", "zip"],
        "computed_columns": {
            "cert_duration_days": {
                "expr": "date_diff('day', completion_date, valid_until)",
                "type": "numeric",
                "null_filter": "completion_date IS NOT NULL AND valid_until IS NOT NULL",
            },
        },
        "temporal_group_map": {
            "year": ("completion_date", "year"),
            "month": ("completion_date", "month"),
            "quarter": ("completion_date", "quarter"),
        },
    },
    "crime_2022": {
        "table": "crime_2022",
        "date_column": "dateend",
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "code_defined", "arrest", "neighborhood", "zip",
            "month",
        ],
        "allowed_filters": {
            "year": "int", "code_defined": "text",
            "neighborhood": "text", "zip": "text",
        },
        "allowed_distinct_columns": ["address", "neighborhood", "zip"],
        "computed_columns": {},
        "temporal_group_map": {
            "month": ("dateend", "month"),
        },
    },
    "rental_registry": {
        "table": "rental_registry",
        "date_column": "completion_date",
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip", "completion_type_name", "rrisvalid",
            "year", "month",
        ],
        "allowed_filters": {
            "year": "int", "zip": "text", "rrisvalid": "text",
        },
        "allowed_distinct_columns": ["sbl", "propertyaddress", "zip"],
        "computed_columns": {},
        "temporal_group_map": {
            "year": ("completion_date", "year"),
            "month": ("completion_date", "month"),
        },
    },
    "unfit_properties": {
        "table": "unfit_properties",
        "date_column": "violation_date",
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip", "status_type_name", "violation", "vacant",
            "year", "month",
        ],
        "allowed_filters": {
            "year": "int", "zip": "text", "status_type_name": "text", "vacant": "text",
        },
        "allowed_distinct_columns": ["sbl", "address"],
        "computed_columns": {},
        "temporal_group_map": {
            "year": ("violation_date", "year"),
            "month": ("violation_date", "month"),
        },
    },
    "trash_pickup": {
        "table": "trash_pickup",
        "date_column": None,
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip", "sanitation", "recyclingw",
        ],
        "allowed_filters": {
            "zip": "text", "sanitation": "text",
        },
        "allowed_distinct_columns": ["sbl"],
        "computed_columns": {},
        "temporal_group_map": {},
    },
    "historical_properties": {
        "table": "historical_properties",
        "date_column": None,
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip", "lpss", "nr_eligible",
        ],
        "allowed_filters": {
            "zip": "text", "lpss": "text", "nr_eligible": "text",
        },
        "allowed_distinct_columns": ["sbl", "property_address"],
        "computed_columns": {},
        "temporal_group_map": {},
    },
    "assessment_roll": {
        "table": "assessment_roll",
        "date_column": None,
        "allowed_metrics": ["count", "count_distinct", "avg", "min", "max"],
        "allowed_group_by": [
            "prop_class_description", "property_city",
        ],
        "allowed_filters": {
            "property_class": "text", "prop_class_description": "text",
        },
        "allowed_distinct_columns": ["sbl", "property_address"],
        "computed_columns": {
            "total_assessment": {
                "expr": "total_assessment",
                "type": "numeric",
                "null_filter": "total_assessment IS NOT NULL",
            },
        },
        "temporal_group_map": {},
    },
    "cityline_requests": {
        "table": "cityline_requests",
        "date_column": "created_at_local",
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip", "category", "agency_name",
            "year", "month",
        ],
        "allowed_filters": {
            "year": "int", "zip": "text", "category": "text", "agency_name": "text",
        },
        "allowed_distinct_columns": ["id", "address"],
        "computed_columns": {},
        "temporal_group_map": {
            "year": ("created_at_local", "year"),
            "month": ("created_at_local", "month"),
        },
    },
    "snow_routes": {
        "table": "snow_routes",
        "date_column": None,
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip",
        ],
        "allowed_filters": {
            "zip": "text",
        },
        "allowed_distinct_columns": ["streetname"],
        "computed_columns": {},
        "temporal_group_map": {},
    },
    "bike_suitability": {
        "table": "bike_suitability",
        "date_column": None,
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "bike_suitability_19",
        ],
        "allowed_filters": {
            "bike_suitability_19": "text",
        },
        "allowed_distinct_columns": ["name"],
        "computed_columns": {},
        "temporal_group_map": {},
    },
    "bike_infrastructure": {
        "table": "bike_infrastructure",
        "date_column": None,
        "allowed_metrics": ["count", "count_distinct", "sum"],
        "allowed_group_by": [
            "infrastructure_type",
        ],
        "allowed_filters": {
            "infrastructure_type": "text",
        },
        "allowed_distinct_columns": ["trail_name"],
        "computed_columns": {
            "length_mi": {
                "expr": "length_mi",
                "type": "numeric",
                "null_filter": "length_mi IS NOT NULL",
            },
        },
        "temporal_group_map": {},
    },
    "parking_violations": {
        "table": "parking_violations",
        "date_column": "issued_date",
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip", "description", "status",
            "year", "month",
        ],
        "allowed_filters": {
            "year": "int", "zip": "text", "description": "text", "status": "text",
        },
        "allowed_distinct_columns": ["ticket_number", "location"],
        "computed_columns": {},
        "temporal_group_map": {
            "year": ("issued_date", "year"),
            "month": ("issued_date", "month"),
        },
    },
    "permit_requests": {
        "table": "permit_requests",
        "date_column": "issue_date",
        "allowed_metrics": ["count", "count_distinct"],
        "allowed_group_by": [
            "zip", "permit_type",
            "year", "month",
        ],
        "allowed_filters": {
            "year": "int", "zip": "text", "permit_type": "text",
        },
        "allowed_distinct_columns": ["permit_number", "full_address"],
        "computed_columns": {},
        "temporal_group_map": {
            "year": ("issue_date", "year"),
            "month": ("issue_date", "month"),
        },
    },
    "tree_inventory": {
        "table": "tree_inventory",
        "date_column": None,
        "allowed_metrics": ["count", "count_distinct", "avg", "min", "max"],
        "allowed_group_by": [
            "zip", "area", "spp_com",
        ],
        "allowed_filters": {
            "zip": "text", "area": "text", "spp_com": "text",
        },
        "allowed_distinct_columns": ["id"],
        "computed_columns": {
            "dbh": {
                "expr": "dbh",
                "type": "numeric",
                "null_filter": "dbh IS NOT NULL",
            },
        },
        "temporal_group_map": {},
    },
    "lead_testing": {
        "table": "lead_testing",
        "date_column": None,
        "allowed_metrics": ["count", "avg", "min", "max"],
        "allowed_group_by": [
            "census_tract", "year",
        ],
        "allowed_filters": {
            "census_tract": "text", "year": "int",
        },
        "allowed_distinct_columns": [],
        "computed_columns": {
            "pct_elevated": {
                "expr": "pct_elevated",
                "type": "numeric",
                "null_filter": "pct_elevated IS NOT NULL",
            },
        },
        "temporal_group_map": {},
    },
}

DATASET_ALIASES = {
    "code violations": "violations",
    "code_violations": "violations",
    "violations": "violations",
    "vacant": "vacant_properties",
    "vacancy": "vacant_properties",
    "vacant properties": "vacant_properties",
    "crime": "crime_2022",
    "crime data": "crime_2022",
    "crime_data": "crime_2022",
    "crimes": "crime_2022",
    "part 1 crime": "crime_2022",
    "rental": "rental_registry",
    "rental registry": "rental_registry",
    "unfit": "unfit_properties",
    "unfit properties": "unfit_properties",
    "trash": "trash_pickup",
    "trash pickup": "trash_pickup",
    "garbage": "trash_pickup",
    "sanitation": "trash_pickup",
    "historical": "historical_properties",
    "historical properties": "historical_properties",
    "landmark": "historical_properties",
    "assessment": "assessment_roll",
    "assessment roll": "assessment_roll",
    "property assessment": "assessment_roll",
    "assessments": "assessment_roll",
    "cityline": "cityline_requests",
    "cityline requests": "cityline_requests",
    "service requests": "cityline_requests",
    "311": "cityline_requests",
    "syrcityline": "cityline_requests",
    "snow routes": "snow_routes",
    "snow": "snow_routes",
    "bike suitability": "bike_suitability",
    "bike infrastructure": "bike_infrastructure",
    "bike lanes": "bike_infrastructure",
    "bike trails": "bike_infrastructure",
    "parking": "parking_violations",
    "parking violations": "parking_violations",
    "parking tickets": "parking_violations",
    "permits": "permit_requests",
    "permit requests": "permit_requests",
    "building permits": "permit_requests",
    "trees": "tree_inventory",
    "tree inventory": "tree_inventory",
    "tree": "tree_inventory",
    "lead": "lead_testing",
    "lead testing": "lead_testing",
    "lead poisoning": "lead_testing",
}

GROUP_BY_ALIASES = {
    "crime_type": "code_defined",
    "zip_code": "zip",
}

# Allowed cross-dataset joins
ALLOWED_JOINS: Dict[tuple, Dict[str, Any]] = {
    ("violations", "rental_registry"): {
        "join_keys": [
            {"left": "complaint_zip", "right": "zip", "type": "zip"},
            {"left": "sbl", "right": "sbl", "type": "sbl"},
        ],
        "description": "Join violations to rental properties",
    },
    ("violations", "vacant_properties"): {
        "join_keys": [
            {"left": "complaint_zip", "right": "zip", "type": "zip"},
            {"left": "sbl", "right": "sbl", "type": "sbl"},
        ],
        "description": "Join violations to vacant properties",
    },
    ("rental_registry", "vacant_properties"): {
        "join_keys": [
            {"left": "zip", "right": "zip", "type": "zip"},
            {"left": "sbl", "right": "sbl", "type": "sbl"},
        ],
        "description": "Join rental registry to vacant properties",
    },
    ("crime_2022", "violations"): {
        "join_keys": [
            {"left": "zip", "right": "complaint_zip", "type": "zip"},
            {"left": "neighborhood", "right": "neighborhood", "type": "neighborhood"},
        ],
        "description": "Join crime data to code violations",
    },
    ("crime_2022", "vacant_properties"): {
        "join_keys": [
            {"left": "zip", "right": "zip", "type": "zip"},
            {"left": "neighborhood", "right": "neighborhood", "type": "neighborhood"},
        ],
        "description": "Join crime data to vacant properties",
    },
    ("unfit_properties", "violations"): {
        "join_keys": [
            {"left": "zip", "right": "complaint_zip", "type": "zip"},
            {"left": "sbl", "right": "sbl", "type": "sbl"},
        ],
        "description": "Join unfit properties to code violations",
    },
    ("unfit_properties", "vacant_properties"): {
        "join_keys": [
            {"left": "zip", "right": "zip", "type": "zip"},
            {"left": "sbl", "right": "sbl", "type": "sbl"},
        ],
        "description": "Join unfit properties to vacant properties",
    },
}

# Aliases for join type detection
JOIN_TYPE_ALIASES = {
    "zip": "zip",
    "zip_code": "zip",
    "zipcode": "zip",
    "sbl": "sbl",
    "property": "sbl",
    "address": "sbl",
    "neighborhood": "neighborhood",
    "neighbourhood": "neighborhood",
}

# Supported filter operators
FILTER_OPS = {"=", ">=", "<=", "between", "in", "like"}


def _normalize_filter(key: str, value: Any, expected_type: str) -> Dict[str, Any]:
    """Normalize a filter value to the expanded {op, value} format."""
    # Already in expanded format
    if isinstance(value, dict) and "op" in value:
        op = value["op"]
        val = value["value"]
        if op not in FILTER_OPS:
            raise ValueError(f"Unsupported filter operator '{op}' for '{key}'.")
        if expected_type == "int":
            if op == "between":
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError(f"Filter '{key}' with 'between' requires [min, max].")
                val = [int(val[0]), int(val[1])]
            elif op == "in":
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"Filter '{key}' with 'in' requires a list.")
                val = [int(v) for v in val]
            else:
                val = int(val)
        else:
            if op == "in":
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"Filter '{key}' with 'in' requires a list.")
                val = [str(v) for v in val]
            elif op != "between":
                val = str(val)
        return {"op": op, "value": val}

    # Old shorthand format: bare value means equality
    if expected_type == "int":
        try:
            return {"op": "=", "value": int(value)}
        except Exception as exc:
            raise ValueError(f"Filter '{key}' must be an integer.") from exc
    else:
        return {"op": "=", "value": str(value)}


def _normalize_group_by(group_by_raw: Any) -> Optional[List[str]]:
    """Normalize group_by to a list of strings or None."""
    if group_by_raw is None:
        return None
    if isinstance(group_by_raw, list):
        result = []
        for g in group_by_raw:
            key = str(g).strip().lower()
            result.append(GROUP_BY_ALIASES.get(key, key))
        return result if result else None
    # Single string
    key = str(group_by_raw).strip().lower()
    resolved = GROUP_BY_ALIASES.get(key, key)
    return [resolved]


def validate_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize structured intent against allowed datasets and fields."""
    if not isinstance(intent, dict):
        raise ValueError("Intent must be a dictionary.")

    dataset_raw = intent.get("dataset")
    dataset_key = str(dataset_raw).strip().lower() if dataset_raw is not None else None
    dataset = DATASET_ALIASES.get(dataset_key, dataset_key)
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_raw}")

    cfg = DATASETS[dataset]

    # Validate metric
    metric = intent.get("metric", "count")
    allowed_metrics = cfg.get("allowed_metrics", ["count"])
    if metric not in allowed_metrics:
        raise ValueError(f"Unsupported metric '{metric}' for dataset '{dataset}'. Allowed: {allowed_metrics}")

    # Validate metric_column (for avg/min/max/sum on computed columns)
    metric_column = intent.get("metric_column")
    if metric in ("avg", "min", "max", "sum"):
        if not metric_column:
            raise ValueError(f"Metric '{metric}' requires a 'metric_column' (computed column).")
        computed = cfg.get("computed_columns", {})
        if metric_column not in computed:
            raise ValueError(f"Unknown computed column '{metric_column}' for dataset '{dataset}'. "
                             f"Available: {list(computed.keys())}")

    # Validate distinct_column (for count_distinct)
    distinct_column = intent.get("distinct_column")
    if metric == "count_distinct":
        if not distinct_column:
            raise ValueError("Metric 'count_distinct' requires a 'distinct_column'.")
        allowed_distinct = cfg.get("allowed_distinct_columns", [])
        if distinct_column not in allowed_distinct:
            raise ValueError(f"Unsupported distinct column '{distinct_column}' for dataset '{dataset}'. "
                             f"Available: {allowed_distinct}")

    # Validate group_by (now supports lists)
    group_by = _normalize_group_by(intent.get("group_by"))
    if group_by is not None:
        for gb in group_by:
            if gb not in cfg["allowed_group_by"]:
                raise ValueError(f"Unsupported group_by '{gb}' for dataset '{dataset}'.")

    # Validate filters (supports both old and new format)
    filters_in = intent.get("filters") or {}
    if not isinstance(filters_in, dict):
        raise ValueError("filters must be an object/dict.")
    filters: Dict[str, Any] = {}
    for key, value in filters_in.items():
        key_norm = str(key).strip().lower()
        allowed_filters = cfg["allowed_filters"]
        if key_norm not in allowed_filters:
            raise ValueError(f"Unsupported filter '{key}' for dataset '{dataset}'.")
        expected_type = allowed_filters[key_norm]
        filters[key_norm] = _normalize_filter(key_norm, value, expected_type)

    # Validate having clause
    having = intent.get("having")
    if having is not None:
        if not isinstance(having, dict) or "op" not in having or "value" not in having:
            raise ValueError("'having' must be {\"op\": \">\", \"value\": 100}.")
        if having["op"] not in ("=", ">", ">=", "<", "<="):
            raise ValueError(f"Unsupported having operator '{having['op']}'.")
        try:
            having = {"op": having["op"], "value": int(having["value"])}
        except (ValueError, TypeError) as exc:
            raise ValueError("'having' value must be a number.") from exc

    limit = intent.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer if provided.")

    return {
        "dataset": dataset,
        "metric": metric,
        "metric_column": metric_column,
        "distinct_column": distinct_column,
        "group_by": group_by,
        "filters": filters,
        "having": having,
        "limit": limit,
    }


def get_dataset_config(dataset: str) -> Dict[str, Any]:
    """Return dataset configuration by canonical name."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASETS[dataset]


def get_join_config(primary: str, secondary: str) -> Dict[str, Any]:
    """Return join configuration for a dataset pair, checking both orderings."""
    key = (primary, secondary)
    if key in ALLOWED_JOINS:
        return {"config": ALLOWED_JOINS[key], "order": "normal"}
    # Check reverse order
    reverse_key = (secondary, primary)
    if reverse_key in ALLOWED_JOINS:
        return {"config": ALLOWED_JOINS[reverse_key], "order": "reversed"}
    raise ValueError(f"No join allowed between '{primary}' and '{secondary}'.")


def validate_join_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a join intent against allowed joins and fields."""
    if not isinstance(intent, dict):
        raise ValueError("Intent must be a dictionary.")

    query_type = intent.get("query_type")
    if query_type != "join":
        raise ValueError("Expected query_type 'join' for join intent.")

    primary_raw = intent.get("primary_dataset")
    primary_key = str(primary_raw).strip().lower() if primary_raw else None
    primary = DATASET_ALIASES.get(primary_key, primary_key)
    if primary not in DATASETS:
        raise ValueError(f"Unsupported primary dataset: {primary_raw}")

    secondary_raw = intent.get("secondary_dataset")
    secondary_key = str(secondary_raw).strip().lower() if secondary_raw else None
    secondary = DATASET_ALIASES.get(secondary_key, secondary_key)
    if secondary not in DATASETS:
        raise ValueError(f"Unsupported secondary dataset: {secondary_raw}")

    if primary == secondary:
        raise ValueError("Primary and secondary datasets must be different.")

    join_info = get_join_config(primary, secondary)
    join_config = join_info["config"]

    join_type_raw = intent.get("join_type", "zip")
    join_type_key = str(join_type_raw).strip().lower()
    join_type = JOIN_TYPE_ALIASES.get(join_type_key, join_type_key)

    valid_join_types = [jk["type"] for jk in join_config["join_keys"]]
    if join_type not in valid_join_types:
        raise ValueError(f"Join type '{join_type_raw}' not supported for {primary}-{secondary}. "
                         f"Allowed: {valid_join_types}")

    # Validate metric (expanded beyond just "count")
    metric = intent.get("metric", "count")
    allowed_metrics = DATASETS[primary].get("allowed_metrics", ["count"])
    if metric not in allowed_metrics:
        raise ValueError(f"Unsupported metric '{metric}' for join with primary '{primary}'. Allowed: {allowed_metrics}")

    # For joins, only count is currently supported in SQL builder
    # (avg/min/max on joins would require more complex CTEs)
    if metric not in ("count", "count_distinct"):
        raise ValueError(f"Metric '{metric}' not yet supported for join queries. Use 'count' or 'count_distinct'.")

    # Validate group_by against primary dataset
    group_by_raw = intent.get("group_by")
    group_by = None
    if group_by_raw is not None:
        group_by_key = str(group_by_raw).strip().lower() if isinstance(group_by_raw, str) else None
        if group_by_key:
            group_by = GROUP_BY_ALIASES.get(group_by_key, group_by_key)
            if group_by not in DATASETS[primary]["allowed_group_by"]:
                join_key_info = next((jk for jk in join_config["join_keys"] if jk["type"] == join_type), None)
                if join_key_info:
                    if join_info["order"] == "normal":
                        allowed_join_field = join_key_info["left"]
                    else:
                        allowed_join_field = join_key_info["right"]
                    if group_by != allowed_join_field and group_by not in DATASETS[primary]["allowed_group_by"]:
                        raise ValueError(f"Unsupported group_by '{group_by_raw}' for primary dataset '{primary}'.")

    # Validate filters (only for primary dataset in join queries)
    filters_in = intent.get("filters") or {}
    if not isinstance(filters_in, dict):
        raise ValueError("filters must be an object/dict.")
    filters: Dict[str, Any] = {}
    for key, value in filters_in.items():
        key_norm = str(key).strip().lower()
        allowed_filters = DATASETS[primary]["allowed_filters"]
        if key_norm not in allowed_filters:
            raise ValueError(f"Unsupported filter '{key}' for dataset '{primary}'.")
        expected_type = allowed_filters[key_norm]
        filters[key_norm] = _normalize_filter(key_norm, value, expected_type)

    limit = intent.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer if provided.")

    return {
        "query_type": "join",
        "primary_dataset": primary,
        "secondary_dataset": secondary,
        "join_type": join_type,
        "metric": metric,
        "group_by": group_by,
        "filters": filters,
        "limit": limit,
    }


__all__ = [
    "DATASETS",
    "DATASET_ALIASES",
    "GROUP_BY_ALIASES",
    "ALLOWED_JOINS",
    "JOIN_TYPE_ALIASES",
    "validate_intent",
    "validate_join_intent",
    "get_dataset_config",
    "get_join_config",
]
