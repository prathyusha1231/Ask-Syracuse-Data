"""
Schema and guardrails for Ask Syracuse Data.
Defines allowed datasets, fields, and validates intents.
"""
from __future__ import annotations
from typing import Dict, Any

DATASETS: Dict[str, Dict[str, Any]] = {
    "violations": {
        "table": "violations",
        "date_column": "violation_date",
        "allowed_group_by": ["neighborhood", "complaint_zip", "status_type_name", "violation"],
        "allowed_filters": {"year": "int", "neighborhood": "text", "complaint_zip": "text", "status_type_name": "text"},
    },
    "vacant_properties": {
        "table": "vacant_properties",
        "date_column": "completion_date",
        "allowed_group_by": ["neighborhood", "zip", "vpr_valid", "vpr_result"],
        "allowed_filters": {"year": "int", "neighborhood": "text", "zip": "text", "vpr_valid": "text"},
    },
    "crime_2022": {
        "table": "crime_2022",
        "date_column": "dateend",
        "allowed_group_by": ["code_defined", "arrest", "neighborhood"],
        "allowed_filters": {"year": "int", "code_defined": "text", "neighborhood": "text"},
    },
    "rental_registry": {
        "table": "rental_registry",
        "date_column": "completion_date",
        "allowed_group_by": ["zip", "completion_type_name", "rrisvalid"],
        "allowed_filters": {"year": "int", "zip": "text", "rrisvalid": "text"},
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
}

GROUP_BY_ALIASES = {
    "crime_type": "code_defined",
    "zip_code": "zip",
}

# Allowed cross-dataset joins with both zip and SBL keys
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
}

# Aliases for join type detection
JOIN_TYPE_ALIASES = {
    "zip": "zip",
    "zip_code": "zip",
    "zipcode": "zip",
    "sbl": "sbl",
    "property": "sbl",
    "address": "sbl",
}


def validate_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize structured intent against allowed datasets and fields."""
    if not isinstance(intent, dict):
        raise ValueError("Intent must be a dictionary.")

    dataset_raw = intent.get("dataset")
    dataset_key = str(dataset_raw).strip().lower() if dataset_raw is not None else None
    dataset = DATASET_ALIASES.get(dataset_key, dataset_key)
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_raw}")

    metric = intent.get("metric")
    if metric != "count":
        raise ValueError("Only 'count' metric is supported.")

    group_by_raw = intent.get("group_by")
    group_by = None
    if group_by_raw is not None:
        group_by_key = str(group_by_raw).strip().lower()
        group_by = GROUP_BY_ALIASES.get(group_by_key, group_by_key)
        if group_by not in DATASETS[dataset]["allowed_group_by"]:
            raise ValueError(f"Unsupported group_by '{group_by_raw}' for dataset '{dataset}'.")

    filters_in = intent.get("filters") or {}
    if not isinstance(filters_in, dict):
        raise ValueError("filters must be an object/dict.")
    filters: Dict[str, Any] = {}
    for key, value in filters_in.items():
        key_norm = str(key).strip().lower()
        allowed_filters = DATASETS[dataset]["allowed_filters"]
        if key_norm not in allowed_filters:
            raise ValueError(f"Unsupported filter '{key}' for dataset '{dataset}'.")
        expected_type = allowed_filters[key_norm]
        if expected_type == "int":
            try:
                filters[key_norm] = int(value)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Filter '{key}' must be an integer.") from exc
        else:
            filters[key_norm] = str(value)

    limit = intent.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer if provided.")

    return {
        "dataset": dataset,
        "metric": metric,
        "group_by": group_by,
        "filters": filters,
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

    # Validate query_type
    query_type = intent.get("query_type")
    if query_type != "join":
        raise ValueError("Expected query_type 'join' for join intent.")

    # Validate and normalize primary dataset
    primary_raw = intent.get("primary_dataset")
    primary_key = str(primary_raw).strip().lower() if primary_raw else None
    primary = DATASET_ALIASES.get(primary_key, primary_key)
    if primary not in DATASETS:
        raise ValueError(f"Unsupported primary dataset: {primary_raw}")

    # Validate and normalize secondary dataset
    secondary_raw = intent.get("secondary_dataset")
    secondary_key = str(secondary_raw).strip().lower() if secondary_raw else None
    secondary = DATASET_ALIASES.get(secondary_key, secondary_key)
    if secondary not in DATASETS:
        raise ValueError(f"Unsupported secondary dataset: {secondary_raw}")

    # Ensure datasets are different
    if primary == secondary:
        raise ValueError("Primary and secondary datasets must be different.")

    # Validate join is allowed
    join_info = get_join_config(primary, secondary)
    join_config = join_info["config"]

    # Validate and normalize join_type
    join_type_raw = intent.get("join_type", "zip")
    join_type_key = str(join_type_raw).strip().lower()
    join_type = JOIN_TYPE_ALIASES.get(join_type_key, join_type_key)

    # Check join_type is supported for this join
    valid_join_types = [jk["type"] for jk in join_config["join_keys"]]
    if join_type not in valid_join_types:
        raise ValueError(f"Join type '{join_type_raw}' not supported for {primary}-{secondary}. "
                         f"Allowed: {valid_join_types}")

    # Validate metric
    metric = intent.get("metric")
    if metric != "count":
        raise ValueError("Only 'count' metric is supported.")

    # Validate group_by against primary dataset
    group_by_raw = intent.get("group_by")
    group_by = None
    if group_by_raw is not None:
        group_by_key = str(group_by_raw).strip().lower()
        group_by = GROUP_BY_ALIASES.get(group_by_key, group_by_key)
        # For join queries, group_by should be valid for primary dataset
        if group_by not in DATASETS[primary]["allowed_group_by"]:
            # Also check if it's a join key field
            join_key_info = next((jk for jk in join_config["join_keys"] if jk["type"] == join_type), None)
            if join_key_info:
                # Allow grouping by the join key fields
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
        if expected_type == "int":
            try:
                filters[key_norm] = int(value)
            except Exception as exc:
                raise ValueError(f"Filter '{key}' must be an integer.") from exc
        else:
            filters[key_norm] = str(value)

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
