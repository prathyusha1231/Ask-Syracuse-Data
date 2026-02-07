"""
Deterministic data loading and light cleaning for Ask Syracuse Data.
All loaders read static CSV snapshots from data/raw and normalize column names.
Includes data quality handling for null values.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Sequence, Dict, Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


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
    df = _apply_null_handling(df, "violations")
    df = _normalize_neighborhood(df)
    return df


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
    df = _apply_null_handling(df, "vacant_properties")
    df = _normalize_neighborhood(df)
    return df


SYRACUSE_ZIP_CENTROIDS = {
    "13202": (43.0410, -76.1489),
    "13203": (43.0607, -76.1369),
    "13204": (43.0444, -76.1758),
    "13205": (43.0123, -76.1452),
    "13206": (43.0677, -76.1102),
    "13207": (43.0195, -76.1650),
    "13208": (43.0730, -76.1486),
    "13210": (43.0354, -76.1282),
    "13214": (43.0397, -76.0722),
    "13215": (42.9722, -76.2276),
    "13219": (43.0409, -76.2262),
    "13224": (43.0421, -76.1046),
}


def _assign_zip_from_coords(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """Derive ZIP codes from lat/long using nearest Syracuse ZIP centroid."""
    df = df.copy()
    zips = list(SYRACUSE_ZIP_CENTROIDS.keys())
    centroids = np.array(list(SYRACUSE_ZIP_CENTROIDS.values()))

    has_coords = df[lat_col].notna() & df[lon_col].notna()
    lats = df.loc[has_coords, lat_col].values
    lons = df.loc[has_coords, lon_col].values

    # Compute distances to each centroid (Euclidean on lat/lon is fine for a single city)
    coords = np.column_stack([lats, lons])  # (N, 2)
    dists = np.sqrt(((coords[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    nearest = dists.argmin(axis=1)
    assigned = [zips[i] for i in nearest]

    df["zip"] = np.nan
    df.loc[has_coords, "zip"] = assigned
    return df


NEIGHBORHOOD_ALIASES = {
    "hawley-green": "Hawley Green",
    "hawley green": "Hawley Green",
    "near westside": "Near Westside",
    "near west side": "Near Westside",
    "far westside": "Far Westside",
    "far west side": "Far Westside",
    "north side": "Northside",
    "northside": "Northside",
    "south side": "Southside",
    "southside": "Southside",
    "east side": "Eastside",
    "eastside": "Eastside",
    "west side": "Westside",
    "westside": "Westside",
    "salt springs": "Salt Springs",
    "sedgwick": "Sedgwick",
    "strathmore": "Strathmore",
    "tipperary hill": "Tipperary Hill",
    "university hill": "University Hill",
    "university neighborhood": "University Neighborhood",
    "downtown": "Downtown",
    "lincoln hill": "Lincoln Hill",
    "meadowbrook": "Meadowbrook",
    "outer comstock": "Outer Comstock",
    "park ave": "Park Ave",
    "prospect hill": "Prospect Hill",
    "skunk city": "Skunk City",
    "south valley": "South Valley",
    "winkworth": "Winkworth",
    "brighton": "Brighton",
    "court-woodlawn": "Court-Woodlawn",
    "court woodlawn": "Court-Woodlawn",
    "elmwood": "Elmwood",
    "lakefront": "Lakefront",
}


def _normalize_neighborhood(df: pd.DataFrame, col: str = "neighborhood") -> pd.DataFrame:
    """Standardize neighborhood names across datasets."""
    if col not in df.columns:
        return df
    df = df.copy()
    lowered = df[col].astype(str).str.strip().str.lower()
    df[col] = lowered.map(NEIGHBORHOOD_ALIASES).fillna(df[col].str.strip())
    return df


def load_crime_2022() -> pd.DataFrame:
    """Load Part 1 crime incidents for 2022 (enriched with geocoded neighborhoods)."""
    # Use enriched file with neighborhood data if available
    enriched_file = DATA_DIR / "Crime_Data_2022_enriched.csv"
    if enriched_file.exists():
        df = _load_csv("Crime_Data_2022_enriched.csv", date_cols=["dateend"])
    else:
        df = _load_csv("Crime_Data_2022_(Part_1_Offenses).csv", date_cols=["dateend"])
    df = _apply_null_handling(df, "crime_2022")
    # Derive ZIP from lat/long (crime data has ~100% null zip but ~97% coords)
    if "latitude" in df.columns and "longitude" in df.columns:
        df = _assign_zip_from_coords(df)
    # Normalize neighborhood names for cross-dataset joins
    df = _normalize_neighborhood(df)
    return df


__all__ = [
    "load_code_violations",
    "load_rental_registry",
    "load_vacant_properties",
    "load_crime_2022",
]
