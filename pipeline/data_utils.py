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
        "vpr_valid": {"strategy": "label", "label": "Not Recorded"},
        "vpr_result": {"strategy": "label", "label": "Not Recorded"},
    },
    "crime": {
        "code_defined": {"strategy": "label", "label": "Unspecified"},
        "arrest": {"strategy": "label", "label": "No"},
        "neighborhood": {"strategy": "label", "label": "Unknown"},
    },
    "unfit_properties": {
        "department_name": {"strategy": "label", "label": "Not Recorded"},
        "complaint_type_name": {"strategy": "label", "label": "Not Recorded"},
        "status_type_name": {"strategy": "label", "label": "Unknown Status"},
    },
    "trash_pickup": {
        "sanitation": {"strategy": "label", "label": "Not Scheduled"},
        "recyclingw": {"strategy": "label", "label": "Not Scheduled"},
    },
    "historical_properties": {
        "lpss": {"strategy": "label", "label": "Not Designated"},
        "nr_eligible": {"strategy": "label", "label": "Not Evaluated"},
    },
    "assessment_roll": {
        "prop_class_description": {"strategy": "label", "label": "Unclassified"},
    },
    "cityline_requests": {
        "agency_name": {"strategy": "label", "label": "Not Assigned"},
        "category": {"strategy": "label", "label": "Uncategorized"},
        "report_source": {"strategy": "label", "label": "Unknown Source"},
    },
    "parking_violations": {
        "description": {"strategy": "label", "label": "Not Recorded"},
        "status": {"strategy": "label", "label": "Unknown Status"},
    },
    "permit_requests": {
        "permit_type": {"strategy": "label", "label": "Not Recorded"},
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
    # Drop 100% null column (GIS artifact)
    df = df.drop(columns=["shape"], errors="ignore")
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
    assigned = [int(zips[i]) for i in nearest]

    df["zip"] = pd.array([pd.NA] * len(df), dtype=pd.Int64Dtype())
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


CRIME_FILES = [
    # (enriched_csv, raw_csv, part)
    ("Crime_Data_2022_enriched.csv", "Crime_Data_2022_(Part_1_Offenses).csv", 1),
    ("Crime_Data_2023_(Part_1_Offenses)_enriched.csv", "Crime_Data_2023_(Part_1_Offenses).csv", 1),
    ("Crime_Data_2023_(Part_2_Offenses)_enriched.csv", "Crime_Data_2023_(Part_2_Offenses).csv", 2),
    ("Crime_Data_2024_(Part_1_Offenses)_enriched.csv", "Crime_Data_2024_(Part_1_Offenses).csv", 1),
    ("Crime_Data_2024_(Part_2_Offenses)_enriched.csv", "Crime_Data_2024_(Part_2_Offenses).csv", 2),
    ("Crime_Data_2025_(Part_1_Offenses)_enriched.csv", "Crime_Data_2025_(Part_1_Offenses).csv", 1),
]


def load_crime() -> pd.DataFrame:
    """Load Part 1 & Part 2 crime data for all available years (2022-2025).

    Prefers pre-merged crime_merged.csv (with ZIP + neighborhood already computed).
    Falls back to loading individual CSVs and computing on the fly.
    """
    merged_path = DATA_DIR / "crime_merged.csv"
    if merged_path.exists():
        df = _load_csv("crime_merged.csv", date_cols=["dateend"])
        if "dateend" in df.columns:
            df["dateend"] = pd.to_datetime(df["dateend"], errors="coerce", utc=True)
        # Ensure zip is nullable int
        if "zip" in df.columns:
            df["zip"] = pd.to_numeric(df["zip"], errors="coerce").astype(pd.Int64Dtype())
        # Filter out stray pre-2022 records and null dates (data entry artifacts)
        if "dateend" in df.columns:
            df = df[df["dateend"].notna() & (df["dateend"] >= "2022-01-01")]
        df = _apply_null_handling(df, "crime")
        return df

    # Fallback: load individual CSVs and compute ZIP/neighborhood on the fly
    frames = []
    for enriched_name, raw_name, part in CRIME_FILES:
        enriched_path = DATA_DIR / enriched_name
        raw_path = DATA_DIR / raw_name
        if enriched_path.exists():
            df = _load_csv(enriched_name, date_cols=["dateend"])
        elif raw_path.exists():
            df = _load_csv(raw_name, date_cols=["dateend"])
        else:
            continue  # skip missing files

        # Normalize lat/long column names (2023/2024/2025 use 'lat'/'long')
        if "lat" in df.columns and "latitude" not in df.columns:
            df = df.rename(columns={"lat": "latitude"})
        if "long" in df.columns and "longitude" not in df.columns:
            df = df.rename(columns={"long": "longitude"})

        # Add year from dateend
        if "dateend" in df.columns:
            df["year"] = pd.to_datetime(df["dateend"], errors="coerce").dt.year

        # Tag part number
        df["crime_part"] = part

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No crime data CSV files found in data/raw/.")

    df = pd.concat(frames, ignore_index=True)

    # Re-parse dateend after concat (mixed formats across CSVs can produce object dtype)
    if "dateend" in df.columns:
        df["dateend"] = pd.to_datetime(df["dateend"], errors="coerce", utc=True)

    # Filter out stray pre-2022 records and null dates (data entry artifacts)
    if "dateend" in df.columns:
        df = df[df["dateend"].notna() & (df["dateend"] >= "2022-01-01")]

    df = _apply_null_handling(df, "crime")

    # Derive ZIP from lat/long
    if "latitude" in df.columns and "longitude" in df.columns:
        df = _assign_zip_from_coords(df)

    # Normalize neighborhood names for cross-dataset joins
    df = _normalize_neighborhood(df)
    return df


def _parse_epoch_ms(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Parse epoch-millisecond timestamps to datetime."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def load_unfit_properties() -> pd.DataFrame:
    """Load unfit properties; epoch-ms dates, SBL normalized."""
    path = DATA_DIR / "Unfit_Properties.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = _parse_epoch_ms(df, ["violation_date", "comply_by_date", "status_date", "comp_open_date", "comp_close_date"])
    df = _normalize_sbl(df)
    # Drop 100% null column
    df = df.drop(columns=["vacant"], errors="ignore")
    df = _apply_null_handling(df, "unfit_properties")
    return df


def load_trash_pickup() -> pd.DataFrame:
    """Load trash pickup schedule data."""
    path = DATA_DIR / "Trash_Pickup_2025.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = df.drop_duplicates()
    df = _normalize_sbl(df)
    # Normalize zip to Int64
    if "zip" in df.columns:
        df["zip"] = pd.to_numeric(df["zip"], errors="coerce").astype(pd.Int64Dtype())
    df = _apply_null_handling(df, "trash_pickup")
    return df


def load_historical_properties() -> pd.DataFrame:
    """Load historical properties with landmark/NR eligibility data."""
    path = DATA_DIR / "Historical_Properties.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = df.drop_duplicates()
    df = _normalize_sbl(df)
    if "zip" in df.columns:
        df["zip"] = pd.to_numeric(df["zip"], errors="coerce").astype(pd.Int64Dtype())
    df = _apply_null_handling(df, "historical_properties")
    return df


def load_assessment_roll() -> pd.DataFrame:
    """Load assessment roll with property values and classifications."""
    path = DATA_DIR / "Assessment_Roll_2026.csv"
    df = pd.read_csv(path, low_memory=False)
    df = _clean_columns(df)
    df = _normalize_sbl(df)
    if "total_assessment" in df.columns:
        df["total_assessment"] = pd.to_numeric(df["total_assessment"], errors="coerce")
    # Extract ZIP from property_city (e.g. "Syracuse, NY 13205" -> 13205)
    if "property_city" in df.columns:
        extracted = df["property_city"].str.extract(r'(\d{5})\s*$', expand=False)
        df["zip"] = pd.to_numeric(extracted, errors="coerce").astype(pd.Int64Dtype())
    df = _apply_null_handling(df, "assessment_roll")
    return df


def load_cityline_requests() -> pd.DataFrame:
    """Load SYRCityline 311 service requests; derive ZIP from lat/long, extract year."""
    path = DATA_DIR / "SYRCityline_Requests.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = df.drop_duplicates()
    # Parse dates with known format: "01/14/2025 - 11:19AM"
    for col in ["created_at_local", "closed_at_local"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y - %I:%M%p", errors="coerce")
    if "lat" in df.columns and "lng" in df.columns:
        df = df.rename(columns={"lat": "latitude", "lng": "longitude"})
        df = _assign_zip_from_coords(df)
    # Extract year for temporal queries
    if "created_at_local" in df.columns:
        df["year"] = df["created_at_local"].dt.year
    df = _apply_null_handling(df, "cityline_requests")
    return df


def load_snow_routes() -> pd.DataFrame:
    """Load emergency snow route road segments; normalize ZIP from postal columns."""
    path = DATA_DIR / "Emergency_Snow_Routes.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    # Normalize left/right postal codes into a single zip column
    if "leftpostal" in df.columns:
        df["zip"] = pd.to_numeric(df["leftpostal"], errors="coerce").astype(pd.Int64Dtype())
    return df


def load_bike_suitability() -> pd.DataFrame:
    """Load bike suitability ratings by road."""
    path = DATA_DIR / "Bike_Suitability_2020.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    return df


def load_bike_infrastructure() -> pd.DataFrame:
    """Load bike infrastructure (trails, lanes, paths)."""
    path = DATA_DIR / "Bike_Infrastructure_2023.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    if "length_mi" in df.columns:
        df["length_mi"] = pd.to_numeric(df["length_mi"], errors="coerce")
    return df


def load_parking_violations() -> pd.DataFrame:
    """Load parking violations; epoch-ms dates, derive ZIP from lat/long."""
    path = DATA_DIR / "Parking_Violations_2023.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = _parse_epoch_ms(df, ["issued_date"])
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    # Rename coordinate columns for _assign_zip_from_coords
    if "lat" in df.columns and "long" in df.columns:
        df = df.rename(columns={"lat": "latitude", "long": "longitude"})
        df = _assign_zip_from_coords(df)
    # Extract year for temporal queries
    if "issued_date" in df.columns:
        df["year"] = df["issued_date"].dt.year
    df = _apply_null_handling(df, "parking_violations")
    return df


def load_permit_requests() -> pd.DataFrame:
    """Load permit requests; epoch-ms dates, derive ZIP from lat/long."""
    path = DATA_DIR / "Permit_Requests.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = _parse_epoch_ms(df, ["issue_date"])
    if "lat" in df.columns and "long" in df.columns:
        df = df.rename(columns={"lat": "latitude", "long": "longitude"})
        df = _assign_zip_from_coords(df)
    elif "latitude" in df.columns and "longitude" in df.columns:
        df = _assign_zip_from_coords(df)
    # Extract year for temporal queries
    if "issue_date" in df.columns:
        df["year"] = df["issue_date"].dt.year
    df = _apply_null_handling(df, "permit_requests")
    return df


def load_tree_inventory() -> pd.DataFrame:
    """Load tree inventory; derive ZIP from lat/long."""
    path = DATA_DIR / "Tree_Inventory.csv"
    df = pd.read_csv(path)
    df = _clean_columns(df)
    if "latitude" in df.columns and "longitude" in df.columns:
        df = _assign_zip_from_coords(df)
    return df


def load_lead_testing() -> pd.DataFrame:
    """Load lead testing data from Excel files (2013-2019 + 2020-2024).
    Files have 3 header rows; real headers at row 3. Data is wide format
    (census_tract x year columns) which we melt to long format.
    """
    frames = []
    for fname in ["Lead_Testing_2013_2019.xlsx", "Lead_Testing_2020_2024.xlsx"]:
        path = DATA_DIR / fname
        if path.exists():
            xls = pd.read_excel(path, engine="openpyxl", header=3)
            # Convert all column names to strings before cleaning
            # (year columns are integers like 2013, and str.lower() turns them to NaN)
            xls.columns = [str(c) for c in xls.columns]
            xls = _clean_columns(xls)
            # Rename first column to census_tract
            first_col = xls.columns[0]
            xls = xls.rename(columns={first_col: "census_tract"})
            xls = xls.dropna(subset=["census_tract"])
            # Melt year columns to long format
            year_cols = [c for c in xls.columns if c != "census_tract"]
            melted = xls.melt(id_vars=["census_tract"], value_vars=year_cols,
                              var_name="year", value_name="pct_elevated")
            melted["year"] = pd.to_numeric(melted["year"], errors="coerce").astype(pd.Int64Dtype())
            # Convert pct_elevated: "Suppressed" -> NaN, otherwise numeric
            melted["pct_elevated"] = pd.to_numeric(melted["pct_elevated"], errors="coerce")
            frames.append(melted)
    if not frames:
        raise FileNotFoundError("No lead testing Excel files found.")
    df = pd.concat(frames, ignore_index=True)
    df["census_tract"] = df["census_tract"].astype(str).str.strip()
    return df


__all__ = [
    "load_code_violations",
    "load_rental_registry",
    "load_vacant_properties",
    "load_crime",
    "load_unfit_properties",
    "load_trash_pickup",
    "load_historical_properties",
    "load_assessment_roll",
    "load_cityline_requests",
    "load_snow_routes",
    "load_bike_suitability",
    "load_bike_infrastructure",
    "load_parking_violations",
    "load_permit_requests",
    "load_tree_inventory",
    "load_lead_testing",
]
