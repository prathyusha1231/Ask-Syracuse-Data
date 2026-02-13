"""
FastAPI backend for Ask Syracuse Data.
Run locally: uvicorn app:app --reload
Deploy: Render, Vercel, or Fly.io
"""
from __future__ import annotations
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sqlite3
import uuid
import datetime
import pandas as pd
from openai import OpenAI

from pipeline.main import run_query
from pipeline.data_utils import SYRACUSE_ZIP_CENTROIDS
from llm.openai_client import load_api_key


# =============================================================================
# FASTAPI APP
# =============================================================================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Ask Syracuse Data",
    description="Natural language interface for Syracuse Open Data",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.plot.ly; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https://*.basemaps.cartocdn.com https://*.global.ssl.fastly.net; "
            "connect-src 'self' https://*.basemaps.cartocdn.com https://*.global.ssl.fastly.net"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Static files & Templates
_BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory="templates")

# Feedback database
FEEDBACK_DB = Path("data/feedback.db")


def _init_feedback_db():
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        question TEXT NOT NULL,
        rating TEXT NOT NULL CHECK(rating IN ('up','down')),
        comment TEXT,
        sql_query TEXT,
        dataset TEXT
    )""")
    conn.commit()
    conn.close()


_init_feedback_db()


# =============================================================================
# LABELS AND FORMATTING
# =============================================================================
COLUMN_LABELS = {
    "count": "Total Count",
    "count_distinct": "Unique Count",
    "zip": "ZIP Code",
    "sbl": "Property ID (SBL)",
    "neighborhood": "Neighborhood",
    "complaint_zip": "ZIP Code",
    "status_type_name": "Violation Status",
    "violation": "Violation Type",
    "completion_type_name": "Completion Status",
    "rrisvalid": "Registration Valid",
    "vpr_valid": "VPR Valid",
    "vpr_result": "VPR Result",
    "code_defined": "Crime Type",
    "arrest": "Arrest Made",
    "primary_count": "Primary Count",
    "secondary_count": "Secondary Count",
    "year": "Year",
    "month": "Month",
    "quarter": "Quarter",
    "avg_days_to_comply": "Avg Days to Comply",
    "avg_days_open": "Avg Days Open",
    "avg_cert_duration_days": "Avg Cert Duration (Days)",
    "min_days_to_comply": "Min Days to Comply",
    "max_days_to_comply": "Max Days to Comply",
    "min_days_open": "Min Days Open",
    "max_days_open": "Max Days Open",
    # New dataset columns
    "sanitation": "Collection Day",
    "recyclingw": "Recycling Week",
    "lpss": "Landmark Status",
    "nr_eligible": "NR Eligible",
    "prop_class_description": "Property Class",
    "property_class": "Property Class Code",
    "property_city": "City",
    "total_assessment": "Total Assessment",
    "avg_total_assessment": "Avg Assessment",
    "min_total_assessment": "Min Assessment",
    "max_total_assessment": "Max Assessment",
    "category": "Category",
    "agency_name": "Agency",
    "infrastructure_type": "Infrastructure Type",
    "trail_name": "Trail Name",
    "length_mi": "Length (Miles)",
    "sum_length_mi": "Total Miles",
    "bike_suitability_19": "Suitability Rating",
    "description": "Description",
    "status": "Status",
    "amount": "Amount",
    "permit_type": "Permit Type",
    "permit_number": "Permit Number",
    "full_address": "Address",
    "area": "Neighborhood",
    "spp_com": "Species",
    "dbh": "Diameter (DBH)",
    "avg_dbh": "Avg Diameter (DBH)",
    "streetname": "Street Name",
    "ticket_number": "Ticket Number",
    "location": "Location",
    "census_tract": "Census Tract",
    "address": "Address",
    "vacant": "Vacant",
}

# Temporal group columns that indicate time-series data
TEMPORAL_COLUMNS = {"year", "month", "quarter"}

DATASET_LABELS = {
    "violations": "Code Violations",
    "rental_registry": "Rental Properties",
    "vacant_properties": "Vacant Properties",
    "crime": "Crime Data (2022-2025)",
    "unfit_properties": "Unfit Properties",
    "trash_pickup": "Trash Pickup",
    "historical_properties": "Historical Properties",
    "assessment_roll": "Assessment Roll",
    "cityline_requests": "SYRCityline Requests",
    "snow_routes": "Snow Routes",
    "bike_suitability": "Bike Suitability",
    "bike_infrastructure": "Bike Infrastructure",
    "parking_violations": "Parking Violations",
    "permit_requests": "Permit Requests",
    "tree_inventory": "Tree Inventory",
    "lead_testing": "Lead Testing",
}


# =============================================================================
# MAP CONSTANTS
# =============================================================================
ZIP_COORDS = {
    str(z): {"lat": lat, "lon": lon}
    for z, (lat, lon) in SYRACUSE_ZIP_CENTROIDS.items()
}

DATASETS_WITH_COORDS = {
    "crime", "cityline_requests", "parking_violations",
    "permit_requests", "tree_inventory", "unfit_properties",
    "historical_properties",
}

ZIP_GROUP_COLUMNS = {"zip", "complaint_zip"}
NEIGHBORHOOD_GROUP_COLUMNS = {"neighborhood", "area"}

# Mapping of Syracuse ZIP codes to their primary neighborhoods
ZIP_TO_NEIGHBORHOODS = {
    "13202": "Downtown, University Hill",
    "13203": "Northside, Hawley Green, Lincoln Hill",
    "13204": "Near Westside, Far Westside, Tipperary Hill, Skunk City",
    "13205": "Southside, South Valley, Elmwood, Brighton",
    "13206": "Eastside, Meadowbrook, Salt Springs",
    "13207": "South Valley, Strathmore, Winkworth",
    "13208": "Northside, Sedgwick, Court-Woodlawn, Prospect Hill",
    "13210": "University Neighborhood, Outer Comstock",
    "13214": "Westside",
    "13215": "South Valley",
    "13219": "Far Westside, Lakefront",
    "13224": "Eastside, Meadowbrook",
}

TRASH_DAY_COLORS = {
    "Monday": "#2563eb",
    "Tuesday": "#16a34a",
    "Wednesday": "#d97706",
    "Thursday": "#dc2626",
    "Friday": "#7c3aed",
    "Not Scheduled": "#9ca3af",
}

NEIGHBORHOOD_COORDS_PY = {
    "Northside": (43.075, -76.155),
    "Brighton": (43.022, -76.147),
    "Near Westside": (43.045, -76.170),
    "Eastwood": (43.048, -76.100),
    "Washington Square": (43.040, -76.135),
    "Elmwood": (43.025, -76.175),
    "Southside": (43.030, -76.150),
    "Park Ave": (43.050, -76.175),
    "Southwest": (43.020, -76.180),
    "North Valley": (43.065, -76.135),
    "Skunk City": (43.010, -76.200),
    "Lincoln Hill": (43.055, -76.125),
    "Court-Woodlawn": (43.060, -76.145),
    "Strathmore": (43.025, -76.115),
    "Salt Springs": (43.035, -76.095),
    "Downtown": (43.050, -76.150),
    "Westcott": (43.040, -76.120),
    "University Hill": (43.038, -76.135),
    "Hawley Green": (43.055, -76.140),
    "Meadowbrook": (43.070, -76.085),
    "Far Westside": (43.035, -76.195),
    "Tipp Hill": (43.040, -76.185),
    "Sedgwick": (43.055, -76.110),
    "Near Eastside": (43.050, -76.140),
    "Outer Comstock": (43.035, -76.110),
    "South Valley": (43.015, -76.140),
    "University Neighborhood": (43.040, -76.130),
    "Lakefront": (43.080, -76.180),
    "Winkworth": (43.005, -76.150),
    "Franklin Square": (43.055, -76.160),
    "South Campus": (43.032, -76.130),
    "Not Recorded": (43.045, -76.150),
}


def get_readable_column(col: str, metadata: dict = None) -> str:
    """Get human-readable column name."""
    if metadata and metadata.get("query_type") == "join":
        primary = metadata.get("primary_dataset", "")
        secondary = metadata.get("secondary_dataset", "")
        if col == "primary_count":
            return f"{DATASET_LABELS.get(primary, primary)} Count"
        elif col == "secondary_count":
            return f"{DATASET_LABELS.get(secondary, secondary)} Count"
    return COLUMN_LABELS.get(col, col.replace("_", " ").title())


def _is_numeric(val) -> bool:
    """Check if value is numeric (int, float, or numpy numeric)."""
    if isinstance(val, (int, float)):
        return True
    return hasattr(val, 'item') and not isinstance(val, (str, bytes))


def format_number(val):
    """Format numbers with commas."""
    if pd.isna(val):
        return val
    if _is_numeric(val):
        if float(val).is_integer():
            return f"{int(val):,}"
        return f"{val:,.2f}"
    return val


YEAR_COLUMNS = {"year", "month", "quarter"}

def format_value(val, column_name: str = ""):
    """Format any value, handling nulls and empty strings."""
    if pd.isna(val) or val == "" or val is None:
        return "Unknown"
    # ZIP columns should display as plain strings (e.g. "13202", not "13,202")
    if column_name in ZIP_GROUP_COLUMNS and _is_numeric(val):
        return str(int(val))
    # Year/month/quarter columns should display as plain integers (e.g. "2023", not "2,023")
    if column_name in YEAR_COLUMNS and _is_numeric(val):
        return str(int(val))
    if _is_numeric(val):
        return format_number(val)
    return str(val)


def _safe_list(series) -> list:
    """Convert a pandas Series to a JSON-safe list (replace NA with None/0)."""
    result = []
    for v in series:
        if pd.isna(v):
            result.append(None)
        elif hasattr(v, 'item'):  # numpy scalar
            result.append(v.item())
        else:
            result.append(v)
    return result


def generate_description(df: pd.DataFrame, metadata: dict) -> str:
    """Generate result description."""
    query_type = metadata.get("query_type", "single")
    row_count = metadata.get("row_count", len(df))
    metric = metadata.get("metric", "count")

    if query_type == "advanced_sql":
        return f"Advanced query result: {row_count} rows returned."

    if query_type == "join":
        primary = DATASET_LABELS.get(metadata.get("primary_dataset", ""), "primary")
        secondary = DATASET_LABELS.get(metadata.get("secondary_dataset", ""), "secondary")
        join_type = metadata.get("join_type", "")

        if join_type == "zip":
            return f"Comparing {primary} and {secondary} by ZIP Code. Found {row_count} ZIP codes with data."
        elif join_type == "neighborhood":
            return f"Comparing {primary} and {secondary} by Neighborhood. Found {row_count} neighborhoods with data."
        else:
            if row_count == 1 and "count" in df.columns:
                count_val = df["count"].iloc[0]
                return f"Property-level match: {primary} and {secondary}. Found {count_val:,} matching records."
            return f"Property-level match between {primary} and {secondary}. Found {row_count} records."

    # Single-dataset queries
    dataset = DATASET_LABELS.get(metadata.get("dataset", ""), "dataset")
    group_by = metadata.get("group_by")

    # Metric label
    metric_desc = ""
    if metric == "count_distinct":
        metric_desc = "unique count of"
    elif metric in ("avg", "min", "max", "sum"):
        metric_col = metadata.get("metric_column", "")
        col_label = COLUMN_LABELS.get(f"{metric}_{metric_col}", metric_col.replace("_", " "))
        metric_desc = f"{col_label} for"

    if group_by:
        # group_by may be a comma-separated string or a single value
        group_label = ", ".join(
            COLUMN_LABELS.get(g.strip(), g.strip().replace("_", " ").title())
            for g in str(group_by).split(", ")
        )
        if metric_desc:
            return f"{metric_desc} {dataset} grouped by {group_label}. Found {row_count} groups."
        return f"{dataset} grouped by {group_label}. Found {row_count} groups."
    else:
        # Ungrouped total
        count_col = "count"
        if metric == "count_distinct":
            count_col = "count_distinct"
        elif metric in ("avg", "min", "max", "sum"):
            metric_col = metadata.get("metric_column", "")
            count_col = f"{metric}_{metric_col}"

        if row_count == 1 and count_col in df.columns:
            val = df[count_col].iloc[0]
            if isinstance(val, (int, float)):
                if float(val).is_integer():
                    return f"Total {dataset}: {int(val):,} {'unique records' if metric == 'count_distinct' else 'records'}"
                return f"{metric_desc} {dataset}: {val:,.2f}" if metric_desc else f"Total {dataset}: {val:,.2f}"
        return f"{dataset}: {row_count} records"


def generate_insights(df: pd.DataFrame, metadata: dict, question: str) -> str | None:
    """Generate AI insights."""
    api_key = load_api_key()
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)
        data_summary = df.head(15).to_string(index=False)
        total_rows = len(df)

        query_type = metadata.get("query_type", "single")
        if query_type == "join":
            context = f"Primary: {metadata.get('primary_dataset')}, Secondary: {metadata.get('secondary_dataset')}, Join: {metadata.get('join_type')}"
        else:
            context = f"Dataset: {metadata.get('dataset')}"

        # Add ZIP-to-neighborhood context when results are grouped by ZIP
        zip_context = ""
        group_by = metadata.get("group_by") or []
        if isinstance(group_by, str):
            group_by = [group_by]
        has_zip_group = any(g in ZIP_GROUP_COLUMNS for g in group_by)
        if has_zip_group:
            zip_lines = [f"  {z}: {n}" for z, n in ZIP_TO_NEIGHBORHOODS.items()]
            zip_context = "\n\nSyracuse ZIP-to-Neighborhood mapping:\n" + "\n".join(zip_lines) + "\n\nWhen discussing ZIP codes, mention which neighborhoods they cover."

        prompt = f"""Analyze this Syracuse Open Data result and provide 2-3 key insights.

Question: {question}
Context: {context}{zip_context}

Data (first {min(15, total_rows)} of {total_rows} rows):
{data_summary}

Provide concise insights (3-4 sentences) that:
1. Highlight patterns or outliers
2. Explain practical meaning for Syracuse residents
3. Note any caveats

Use bullet points. Don't repeat raw numbers - interpret them."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst helping Syracuse residents understand city data. Be concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Unable to generate insights: {str(e)}"


# =============================================================================
# MAP DATA GENERATION
# =============================================================================
def _find_value_column(df: pd.DataFrame, exclude_col: str) -> str | None:
    """Find the first numeric/metric column that is not the category column."""
    for col in df.columns:
        if col == exclude_col:
            continue
        if col in ("count", "count_distinct", "primary_count", "secondary_count") \
                or col.startswith(("avg_", "min_", "max_", "sum_")):
            return col
    return None


def _build_neighborhood_bubble(df: pd.DataFrame, geo_col: str, value_col: str) -> dict | None:
    """Build neighborhood bubble map data."""
    lats, lons, labels, values = [], [], [], []
    for _, row in df.iterrows():
        name = str(row[geo_col])
        coords = NEIGHBORHOOD_COORDS_PY.get(name)
        if coords:
            lats.append(coords[0])
            lons.append(coords[1])
            labels.append(name)
            values.append(row[value_col] if not pd.isna(row[value_col]) else 0)
    if not lats:
        return None
    return {
        "type": "neighborhood_bubble",
        "lats": lats,
        "lons": lons,
        "labels": labels,
        "values": _safe_list(pd.Series(values)),
        "value_label": get_readable_column(value_col),
    }


def _build_zip_bubble(df: pd.DataFrame, zip_col: str, value_col: str) -> dict | None:
    """Build ZIP-level bubble map data."""
    lats, lons, labels, values = [], [], [], []
    for _, row in df.iterrows():
        z = str(int(row[zip_col])) if not pd.isna(row[zip_col]) else None
        if z and z in ZIP_COORDS:
            c = ZIP_COORDS[z]
            lats.append(c["lat"])
            lons.append(c["lon"])
            labels.append(z)
            values.append(row[value_col] if not pd.isna(row[value_col]) else 0)
    if not lats:
        return None
    return {
        "type": "zip_bubble",
        "lats": lats,
        "lons": lons,
        "labels": labels,
        "values": _safe_list(pd.Series(values)),
        "value_label": get_readable_column(value_col),
    }


def _build_trash_zip_map(raw_df: pd.DataFrame) -> dict | None:
    """Build trash pickup ZIP map with day-colored bubbles from raw data."""
    if "zip" not in raw_df.columns or "sanitation" not in raw_df.columns:
        return None

    # Count addresses per (zip, sanitation day)
    grouped = raw_df.groupby(["zip", "sanitation"]).size().reset_index(name="cnt")

    # For each ZIP, find the dominant day (most addresses)
    idx = grouped.groupby("zip")["cnt"].idxmax()
    dominant = grouped.loc[idx]

    # Total count per ZIP
    zip_totals = raw_df.groupby("zip").size().reset_index(name="total")
    merged = dominant.merge(zip_totals, on="zip", how="left")

    lats, lons, labels, values, colors, days = [], [], [], [], [], []
    for _, row in merged.iterrows():
        z = str(int(row["zip"])) if not pd.isna(row["zip"]) else None
        if not z or z not in ZIP_COORDS:
            continue
        c = ZIP_COORDS[z]
        lats.append(c["lat"])
        lons.append(c["lon"])
        labels.append(z)
        values.append(int(row["total"]))
        day = str(row["sanitation"])
        days.append(day)
        colors.append(TRASH_DAY_COLORS.get(day, "#6b7280"))

    if not lats:
        return None
    return {
        "type": "zip_bubble",
        "lats": lats,
        "lons": lons,
        "labels": labels,
        "values": values,
        "colors": colors,
        "days": days,
        "value_label": "Properties",
        "color_by": "sanitation",
        "color_legend": TRASH_DAY_COLORS,
    }


def _build_point_map(raw_df: pd.DataFrame, dataset: str, limit: int = 2000) -> dict | None:
    """Build a point scatter map from raw row-level lat/long data."""
    valid = raw_df[raw_df["latitude"].notna() & raw_df["longitude"].notna()].head(limit)
    if valid.empty:
        return None

    lats = _safe_list(valid["latitude"])
    lons = _safe_list(valid["longitude"])

    # Build hover text from a meaningful column
    text_col = None
    for candidate in ["address", "location", "full_address", "description",
                       "spp_com", "category", "property_address"]:
        if candidate in valid.columns:
            text_col = candidate
            break

    texts = []
    for _, row in valid.iterrows():
        t = str(row[text_col]) if text_col and not pd.isna(row.get(text_col)) else ""
        texts.append(t)

    return {
        "type": "point",
        "lats": lats,
        "lons": lons,
        "texts": texts,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "point_count": len(valid),
    }


def generate_map_data(df: pd.DataFrame, metadata: dict, raw_df=None) -> dict | None:
    """Generate map visualization data based on query results and metadata."""
    query_type = metadata.get("query_type", "single")
    dataset = metadata.get("dataset", "")
    group_by_str = metadata.get("group_by")

    # Parse group_by to a list
    if group_by_str:
        group_by_list = [g.strip() for g in str(group_by_str).split(",")]
    else:
        group_by_list = []

    # Skip map for multi group-by where first col isn't geographic
    first_group = group_by_list[0] if group_by_list else None

    # Case 1: Neighborhood bubble map
    if first_group and first_group in NEIGHBORHOOD_GROUP_COLUMNS and first_group in df.columns:
        value_col = _find_value_column(df, first_group)
        if value_col:
            return _build_neighborhood_bubble(df, first_group, value_col)

    # Case 2: ZIP bubble map
    zip_col = None
    if first_group and first_group in ZIP_GROUP_COLUMNS:
        zip_col = first_group
    # Also check join queries grouped by zip
    if query_type == "join" and metadata.get("join_type") in ("zip",):
        for col in df.columns:
            if col in ZIP_GROUP_COLUMNS:
                zip_col = col
                break

    if zip_col and zip_col in df.columns:
        # Special case: trash_pickup → show collection day colors
        if dataset == "trash_pickup" and raw_df is not None and "sanitation" in raw_df.columns:
            return _build_trash_zip_map(raw_df)
        value_col = _find_value_column(df, zip_col)
        if value_col:
            return _build_zip_bubble(df, zip_col, value_col)

    # Case 3: Point map for ungrouped queries on datasets with lat/long
    if (not group_by_list
            and dataset in DATASETS_WITH_COORDS
            and raw_df is not None
            and "latitude" in raw_df.columns
            and "longitude" in raw_df.columns):
        return _build_point_map(raw_df, dataset)

    return None


# =============================================================================
# DATA SOURCE CITATIONS
# =============================================================================
DATA_CITATIONS = {
    "violations": {
        "name": "Code Violations V2",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/code-violations",
        "date_range": "2017-present",
        "update_frequency": "Daily",
        "caveats": [
            "Records reflect enforcement activity, not all actual violations",
            "Reporting patterns may vary by neighborhood and inspector",
            "Address-level data may be approximate",
        ],
    },
    "rental_registry": {
        "name": "Syracuse Rental Registry",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/rental-registry",
        "date_range": "Current registrations",
        "update_frequency": "Weekly",
        "caveats": [
            "Only includes registered rental properties",
            "Unregistered rentals are not captured",
            "Registration status may lag actual conditions",
        ],
    },
    "vacant_properties": {
        "name": "Vacant Properties",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/vacant-properties",
        "date_range": "Current listings",
        "update_frequency": "Monthly",
        "caveats": [
            "Administrative designation may not reflect actual occupancy",
            "Properties may be vacant but not yet identified",
            "Demolition/renovation status may be outdated",
        ],
    },
    "crime": {
        "name": "Crime Data (Part 1 & 2 Offenses, 2022-2025)",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov",
        "date_range": "2022-2025 (2025 partial)",
        "update_frequency": "Annual",
        "caveats": [
            "Part 1 (serious) and Part 2 (less serious) offenses included",
            "Addresses generalized to block level for privacy",
            "Reflects reported crimes, not all incidents",
            "Under-reporting varies by crime type",
            "2025 data is partial (~33 records)",
            "Neighborhood data may be unavailable for 2023-2025",
        ],
    },
    "unfit_properties": {
        "name": "Unfit Properties",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/unfit-properties",
        "date_range": "Current listings",
        "update_frequency": "Periodic",
        "caveats": [
            "Properties declared unfit for human habitation",
            "Status may change after listing",
        ],
    },
    "trash_pickup": {
        "name": "Trash Pickup Schedule 2025",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/trash-pickup",
        "date_range": "2025 schedule",
        "update_frequency": "Annual",
        "caveats": [
            "Schedule data only, not collection performance",
            "Holiday adjustments not reflected",
        ],
    },
    "historical_properties": {
        "name": "Historical Properties",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/historical-properties",
        "date_range": "Current listings",
        "update_frequency": "Periodic",
        "caveats": [
            "Includes both designated landmarks and potentially eligible properties",
            "Eligibility status may not reflect current conditions",
        ],
    },
    "assessment_roll": {
        "name": "Assessment Roll 2026",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/assessment-roll",
        "date_range": "2026 assessment year",
        "update_frequency": "Annual",
        "caveats": [
            "Assessed values may differ from market values",
            "Exemptions reduce taxable value but not assessment",
        ],
    },
    "cityline_requests": {
        "name": "SYRCityline Service Requests",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/syrcityline-requests",
        "date_range": "Multi-year",
        "update_frequency": "Daily",
        "caveats": [
            "Reflects reported requests, not all city issues",
            "Response times may not reflect resolution quality",
        ],
    },
    "snow_routes": {
        "name": "Emergency Snow Routes",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/emergency-snow-routes",
        "date_range": "Current routes",
        "update_frequency": "Periodic",
        "caveats": [
            "Road segment data, not event/plow tracking",
            "Routes may change seasonally",
        ],
    },
    "bike_suitability": {
        "name": "Bike Suitability 2020",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/bike-suitability",
        "date_range": "2020 assessment",
        "update_frequency": "Static",
        "caveats": [
            "Ratings from 2020, road conditions may have changed",
            "Does not account for seasonal conditions",
        ],
    },
    "bike_infrastructure": {
        "name": "Bike Infrastructure 2023",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/bike-infrastructure",
        "date_range": "2023",
        "update_frequency": "Annual",
        "caveats": [
            "Only includes officially mapped infrastructure",
            "Condition/maintenance not tracked",
        ],
    },
    "parking_violations": {
        "name": "Parking Violations 2023",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/parking-violations",
        "date_range": "2023",
        "update_frequency": "Static (annual)",
        "caveats": [
            "Reflects issued tickets, not all parking violations",
            "Fine amounts are initial, not adjusted for appeals",
        ],
    },
    "permit_requests": {
        "name": "Permit Requests",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/permit-requests",
        "date_range": "Multi-year",
        "update_frequency": "Daily",
        "caveats": [
            "Reflects permit applications, not completion",
            "Some permits may be denied or withdrawn",
        ],
    },
    "tree_inventory": {
        "name": "Tree Inventory",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/tree-inventory",
        "date_range": "Current inventory",
        "update_frequency": "Periodic",
        "caveats": [
            "Covers city-managed trees, not private trees",
            "Tree health/condition not always current",
        ],
    },
    "lead_testing": {
        "name": "Lead Testing (2013-2024)",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/lead-testing",
        "date_range": "2013-2024",
        "update_frequency": "Annual",
        "caveats": [
            "Census-tract level data, not individual addresses",
            "Screening rates vary by tract and year",
            "Research dataset - interpret with caution",
        ],
    },
}


def get_citation_text(dataset: str) -> dict:
    """Get formatted citation for a dataset."""
    citation = DATA_CITATIONS.get(dataset, {})
    if not citation:
        return {}
    return {
        "name": citation.get("name", dataset),
        "source": citation.get("source", "Syracuse Open Data"),
        "url": citation.get("url", "https://data.syr.gov"),
        "date_range": citation.get("date_range", "Unknown"),
        "caveats": citation.get("caveats", []),
    }


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class QueryResponse(BaseModel):
    success: bool
    query_id: str | None = None
    description: str | None = None
    columns: list[str] | None = None
    data: list[dict] | None = None
    chart_data: dict | None = None
    map_data: dict | None = None
    insights: str | None = None
    sql: str | None = None
    metadata: dict | None = None
    limitations: str | None = None
    validation: dict | None = None
    bias_warnings: list[dict] | None = None
    citations: list[dict] | None = None
    clarification: dict | None = None
    error: str | None = None


class FeedbackRequest(BaseModel):
    query_id: str
    question: str
    rating: str
    comment: str | None = None
    sql: str | None = None
    dataset: str | None = None


# =============================================================================
# ROUTES
# =============================================================================
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files."""
    full_path = _BASE_DIR / "static" / file_path
    if not full_path.is_file() or not full_path.resolve().is_relative_to(_BASE_DIR / "static"):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(full_path)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query_data(request: Request, req: QueryRequest):
    """Process a natural language query."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    query_id = str(uuid.uuid4())
    result = run_query(req.question)

    # Handle location clarification
    if result.get("clarification"):
        clar = result["clarification"]
        dataset = clar.get("detected_dataset", "")
        original_q = clar.get("original_question", req.question)

        # Build ZIP options
        zip_options = [
            {"value": z, "label": z, "detail": n}
            for z, n in ZIP_TO_NEIGHBORHOODS.items()
        ]

        # Build neighborhood options (for datasets that support neighborhood grouping)
        neighborhood_options = []
        datasets_with_neighborhoods = {"crime", "violations", "vacant_properties"}
        if dataset in datasets_with_neighborhoods:
            neighborhood_options = [
                {"value": name, "label": name}
                for name in sorted(NEIGHBORHOOD_COORDS_PY.keys())
                if name not in ("Not Recorded", "Unknown")
            ]

        return QueryResponse(
            success=False,
            query_id=query_id,
            clarification={
                "type": "location",
                "message": "Which area are you asking about? Pick a ZIP code or neighborhood below.",
                "original_question": original_q,
                "dataset": dataset,
                "zip_options": zip_options,
                "neighborhood_options": neighborhood_options,
            },
        )

    if result.get("error"):
        return QueryResponse(
            success=False,
            query_id=query_id,
            error=result["error"],
            sql=result.get("sql"),
            metadata=result.get("metadata"),
            validation=result.get("validation"),
            bias_warnings=result.get("bias_warnings"),
        )

    df = result.get("result")
    metadata = result.get("metadata", {})

    if not isinstance(df, pd.DataFrame) or df.empty:
        q_lower = req.question.lower()
        if any(w in q_lower for w in ["safest", "best", "worst", "should i", "recommend", "which zip should"]):
            error_msg = ("No results returned. This looks like a subjective or recommendation question. "
                        "Try asking about specific data instead, e.g. 'Crime by neighborhood' or 'Violations by zip code'.")
        else:
            error_msg = "No results returned. Try rephrasing your question or checking the dataset name."
        return QueryResponse(
            success=False,
            error=error_msg,
            sql=result.get("sql"),
            metadata=metadata,
        )

    # Push "Unknown" / null / "Not Recorded" label values to bottom of grouped results
    group_by = metadata.get("group_by")
    group_list = [group_by] if isinstance(group_by, str) else (group_by or [])
    if group_list and len(df) > 1:
        first_group = group_list[0]
        if first_group in df.columns:
            unknown_labels = {"Unknown", "Not Recorded", "Not Scheduled", "Unspecified"}
            is_null = df[first_group].isna()
            is_unknown_label = df[first_group].isin(unknown_labels) if df[first_group].dtype == object else pd.Series(False, index=df.index)
            is_bottom = is_null | is_unknown_label
            if is_bottom.any():
                df = pd.concat([df[~is_bottom], df[is_bottom]], ignore_index=True)

    # Format columns
    columns = [get_readable_column(col, metadata) for col in df.columns]

    # Format data
    formatted_data = []
    for _, row in df.iterrows():
        formatted_row = {}
        for col in df.columns:
            readable_col = get_readable_column(col, metadata)
            formatted_row[readable_col] = format_value(row[col], column_name=col)
        formatted_data.append(formatted_row)

    # Prepare chart data
    chart_data = None
    if len(df) >= 2:
        raw_columns = list(df.columns)
        numeric_col_names = {"count", "count_distinct", "primary_count", "secondary_count"}
        # Also match avg_*, min_*, max_*, sum_* columns
        category_col = None
        value_cols = []

        for col in raw_columns:
            if col.lower() in numeric_col_names or col.startswith(("avg_", "min_", "max_", "sum_")):
                value_cols.append(col)
            elif category_col is None:
                category_col = col

        if category_col and value_cols:
            chart_df = df.head(15)

            # Determine chart type
            is_temporal = category_col in TEMPORAL_COLUMNS
            is_join_grouped = (
                metadata.get("query_type") == "join"
                and metadata.get("join_type") in ("zip", "neighborhood")
                and len(value_cols) >= 2
            )

            # For ZIP columns, convert labels to strings so Plotly doesn't format as "13.2K"
            is_zip_category = category_col in ZIP_GROUP_COLUMNS
            if is_zip_category:
                zip_labels = [str(int(v)) if not pd.isna(v) else "Unknown" for v in chart_df[category_col]]

            if is_join_grouped:
                # Grouped bar chart for cross-dataset comparisons
                chart_data = {
                    "type": "grouped_bar",
                    "labels": zip_labels if is_zip_category else _safe_list(chart_df[category_col]),
                    "datasets": [
                        {
                            "label": DATASET_LABELS.get(metadata.get("primary_dataset", ""), "Primary"),
                            "data": _safe_list(chart_df[value_cols[0]]),
                        },
                        {
                            "label": DATASET_LABELS.get(metadata.get("secondary_dataset", ""), "Secondary"),
                            "data": _safe_list(chart_df[value_cols[1]]) if len(value_cols) > 1 else [],
                        }
                    ],
                    "x_label": get_readable_column(category_col, metadata),
                }
            elif is_temporal:
                # Line chart for temporal data
                chart_data = {
                    "type": "line",
                    "labels": [str(v) if not pd.isna(v) else "Unknown" for v in chart_df[category_col]],
                    "values": _safe_list(chart_df[value_cols[0]]),
                    "x_label": get_readable_column(category_col, metadata),
                    "y_label": get_readable_column(value_cols[0], metadata),
                }
            else:
                # Simple bar chart
                chart_data = {
                    "type": "bar",
                    "labels": zip_labels if is_zip_category else _safe_list(chart_df[category_col]),
                    "values": _safe_list(chart_df[value_cols[0]]),
                    "x_label": get_readable_column(category_col, metadata),
                    "y_label": get_readable_column(value_cols[0], metadata),
                }

    # Generate map data
    raw_df = result.get("raw_df")
    map_data = generate_map_data(df, metadata, raw_df=raw_df)

    # Generate description
    description = generate_description(df, metadata)

    # Add note when user asked for neighborhood but data only has ZIP
    if "neighborhood" in req.question.lower():
        group_by = metadata.get("group_by")
        group_list = [group_by] if isinstance(group_by, str) else (group_by or [])
        if any(g in ZIP_GROUP_COLUMNS for g in group_list) and not any(g in NEIGHBORHOOD_GROUP_COLUMNS for g in group_list):
            description += " (Note: neighborhood data not available for this dataset, showing by ZIP code instead.)"

    # Generate insights
    insights = generate_insights(df, metadata, req.question)

    # Get citations based on datasets used
    citations = []
    if metadata.get("query_type") == "advanced_sql":
        # Advanced SQL may use any/all tables — cite all
        for ds in DATA_CITATIONS:
            citations.append(get_citation_text(ds))
    elif metadata.get("query_type") == "join":
        primary = metadata.get("primary_dataset")
        secondary = metadata.get("secondary_dataset")
        if primary:
            citations.append(get_citation_text(primary))
        if secondary:
            citations.append(get_citation_text(secondary))
    else:
        dataset = metadata.get("dataset")
        if dataset:
            citations.append(get_citation_text(dataset))

    return QueryResponse(
        success=True,
        query_id=query_id,
        description=description,
        columns=columns,
        data=formatted_data,
        chart_data=chart_data,
        map_data=map_data,
        insights=insights,
        sql=result.get("sql"),
        metadata=metadata,
        limitations=result.get("limitations"),
        validation=result.get("validation"),
        bias_warnings=result.get("bias_warnings"),
        citations=citations,
    )


@app.post("/api/feedback")
@limiter.limit("30/minute")
async def submit_feedback(request: Request, req: FeedbackRequest):
    """Record user feedback on a query result."""
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'")

    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        conn.execute(
            "INSERT INTO feedback (query_id, timestamp, question, rating, comment, sql_query, dataset) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                req.query_id,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                req.question,
                req.rating,
                req.comment,
                req.sql,
                req.dataset,
            ),
        )
        conn.commit()
        return {"success": True, "message": "Feedback recorded"}
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Failed to record feedback")
    finally:
        conn.close()


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Return feedback statistics."""
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) as n FROM feedback").fetchone()["n"]
        thumbs_up = conn.execute("SELECT COUNT(*) as n FROM feedback WHERE rating='up'").fetchone()["n"]
        thumbs_down = conn.execute("SELECT COUNT(*) as n FROM feedback WHERE rating='down'").fetchone()["n"]
        recent_rows = conn.execute(
            "SELECT query_id, timestamp, question, rating, comment FROM feedback ORDER BY id DESC LIMIT 10"
        ).fetchall()
        recent = [dict(r) for r in recent_rows]
        return {"total": total, "thumbs_up": thumbs_up, "thumbs_down": thumbs_down, "recent": recent}
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback stats")
    finally:
        conn.close()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Ask Syracuse Data"}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
