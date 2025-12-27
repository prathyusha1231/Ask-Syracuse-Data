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
        "allowed_group_by": ["code_defined", "arrest"],
        "allowed_filters": {"year": "int", "code_defined": "text"},
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


__all__ = [
    "DATASETS",
    "DATASET_ALIASES",
    "GROUP_BY_ALIASES",
    "validate_intent",
    "get_dataset_config",
]
