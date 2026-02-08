"""
FastAPI backend for Ask Syracuse Data.
Run locally: uvicorn app:app --reload
Deploy: Render, Vercel, or Fly.io
"""
from __future__ import annotations
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import sqlite3
import uuid
import datetime
import pandas as pd
from openai import OpenAI

from pipeline.main import run_query
from llm.openai_client import load_api_key


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Ask Syracuse Data",
    description="Natural language interface for Syracuse Open Data",
    version="1.0.0"
)

# Templates
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


def format_number(val):
    """Format numbers with commas."""
    if pd.isna(val):
        return val
    if isinstance(val, (int, float)):
        if float(val).is_integer():
            return f"{int(val):,}"
        return f"{val:,.2f}"
    return val


def format_value(val):
    """Format any value, handling nulls and empty strings."""
    if pd.isna(val) or val == "" or val is None:
        return "Unknown"
    if isinstance(val, (int, float)):
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

        prompt = f"""Analyze this Syracuse Open Data result and provide 2-3 key insights.

Question: {question}
Context: {context}

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
    question: str


class QueryResponse(BaseModel):
    success: bool
    query_id: str | None = None
    description: str | None = None
    columns: list[str] | None = None
    data: list[dict] | None = None
    chart_data: dict | None = None
    insights: str | None = None
    sql: str | None = None
    metadata: dict | None = None
    limitations: str | None = None
    validation: dict | None = None
    bias_warnings: list[dict] | None = None
    citations: list[dict] | None = None
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
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query", response_model=QueryResponse)
async def query_data(req: QueryRequest):
    """Process a natural language query."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    query_id = str(uuid.uuid4())
    result = run_query(req.question)

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
        return QueryResponse(
            success=False,
            error="No results returned",
            sql=result.get("sql"),
            metadata=metadata,
        )

    # Format columns
    columns = [get_readable_column(col, metadata) for col in df.columns]

    # Format data
    formatted_data = []
    for _, row in df.iterrows():
        formatted_row = {}
        for col in df.columns:
            readable_col = get_readable_column(col, metadata)
            formatted_row[readable_col] = format_value(row[col])
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

            if is_join_grouped:
                # Grouped bar chart for cross-dataset comparisons
                chart_data = {
                    "type": "grouped_bar",
                    "labels": _safe_list(chart_df[category_col]),
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
                    "labels": _safe_list(chart_df[category_col]),
                    "values": _safe_list(chart_df[value_cols[0]]),
                    "x_label": get_readable_column(category_col, metadata),
                    "y_label": get_readable_column(value_cols[0], metadata),
                }

    # Generate description
    description = generate_description(df, metadata)

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
        insights=insights,
        sql=result.get("sql"),
        metadata=metadata,
        limitations=result.get("limitations"),
        validation=result.get("validation"),
        bias_warnings=result.get("bias_warnings"),
        citations=citations,
    )


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Record user feedback on a query result."""
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'")

    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute(
        "INSERT INTO feedback (query_id, timestamp, question, rating, comment, sql_query, dataset) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            req.query_id,
            datetime.datetime.utcnow().isoformat(),
            req.question,
            req.rating,
            req.comment,
            req.sql,
            req.dataset,
        ),
    )
    conn.commit()
    conn.close()
    return {"success": True, "message": "Feedback recorded"}


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Return feedback statistics."""
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) as n FROM feedback").fetchone()["n"]
    thumbs_up = conn.execute("SELECT COUNT(*) as n FROM feedback WHERE rating='up'").fetchone()["n"]
    thumbs_down = conn.execute("SELECT COUNT(*) as n FROM feedback WHERE rating='down'").fetchone()["n"]
    recent_rows = conn.execute(
        "SELECT query_id, timestamp, question, rating, comment FROM feedback ORDER BY id DESC LIMIT 10"
    ).fetchall()
    conn.close()
    recent = [dict(r) for r in recent_rows]
    return {"total": total, "thumbs_up": thumbs_up, "thumbs_down": thumbs_down, "recent": recent}


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
