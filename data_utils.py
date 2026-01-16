"""
Deterministic data loading and light cleaning for Ask Syracuse Data.
All loaders read static CSV snapshots from data/raw and normalize column names.
Includes data quality handling for null values.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Sequence, Dict, Any

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


# =============================================================================
# NULL HANDLING STRATEGIES
# =============================================================================
NULL_STRATEGIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "violations": {
        "neighborhood": {"strategy": "label", "label": "Not Recorded"},
        "status_type_name": {"strategy": "label", "label": "Unknown Status"},
    },
    "rental_registry": {
        "completion_type_name": {"strategy": "label", "label": "Not Recorded"},
    },
    "vacant_properties": {
        "neighborhood": {"strategy": "label", "label": "Not Recorded"},
    },
    "crime_2022": {
        "code_defined": {"strategy": "label", "label": "Unspecified"},
        "arrest": {"strategy": "label", "label": "Unknown"},
        "neighborhood": {"strategy": "label", "label": "Unknown"},
    },
}


def _apply_null_handling(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply null handling strategies to a dataset."""
    df = df.copy()
    strategies = NULL_STRATEGIES.get(dataset_name, {})

    for col, config in strategies.items():
        if col not in df.columns:
            continue

        strategy = config.get("strategy", "keep")

        if strategy == "label":
            label = config.get("label", "Not Recorded")
            df[col] = df[col].fillna(label)
            if df[col].dtype == 'object':
                df[col] = df[col].replace('', label)

    return df


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def _parse_dates(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _normalize_sbl(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize SBL (Standard Boundary Locator) field for reliable joins."""
    df = df.copy()
    if "sbl" in df.columns:
        df["sbl"] = df["sbl"].astype(str).str.strip().str.upper()
        # Replace 'NAN' strings with empty string for cleaner joins
        df["sbl"] = df["sbl"].replace("NAN", "")
    return df


def _load_csv(filename: str, date_cols: Sequence[str]) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = _parse_dates(df, date_cols)
    return df


def load_code_violations() -> pd.DataFrame:
    """Load housing code violations; columns lowercased, dates parsed, SBL normalized, nulls handled."""
    df = _load_csv(
        "Code_Violations_V2.csv",
        date_cols=["open_date", "violation_date", "status_date", "comply_by_date"],
    )
    df = _normalize_sbl(df)
    return _apply_null_handling(df, "violations")


def load_rental_registry() -> pd.DataFrame:
    """Load rental registry records; columns lowercased, dates parsed, SBL normalized, nulls handled."""
    df = _load_csv(
        "Syracuse_Rental_Registry.csv",
        date_cols=["completion_date", "valid_until"],
    )
    df = _normalize_sbl(df)
    return _apply_null_handling(df, "rental_registry")


def load_vacant_properties() -> pd.DataFrame:
    """Load vacant properties; columns lowercased, dates parsed, SBL normalized, nulls handled."""
    df = _load_csv(
        "Vacant_Properties.csv",
        date_cols=["completion_date", "valid_until"],
    )
    df = _normalize_sbl(df)
    return _apply_null_handling(df, "vacant_properties")


def load_crime_2022() -> pd.DataFrame:
    """Load Part 1 crime incidents for 2022 (enriched with geocoded neighborhoods)."""
    # Use enriched file with neighborhood data if available
    enriched_file = DATA_DIR / "Crime_Data_2022_enriched.csv"
    if enriched_file.exists():
        df = _load_csv("Crime_Data_2022_enriched.csv", date_cols=["dateend"])
    else:
        df = _load_csv("Crime_Data_2022_(Part_1_Offenses).csv", date_cols=["dateend"])
    return _apply_null_handling(df, "crime_2022")


__all__ = [
    "load_code_violations",
    "load_rental_registry",
    "load_vacant_properties",
    "load_crime_2022",
]
