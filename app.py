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


# =============================================================================
# LABELS AND FORMATTING
# =============================================================================
COLUMN_LABELS = {
    "count": "Total Count",
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
}

DATASET_LABELS = {
    "violations": "Code Violations",
    "rental_registry": "Rental Properties",
    "vacant_properties": "Vacant Properties",
    "crime_2022": "Crimes (2022)",
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


def generate_description(df: pd.DataFrame, metadata: dict) -> str:
    """Generate result description."""
    query_type = metadata.get("query_type", "single")
    row_count = metadata.get("row_count", len(df))

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
    else:
        dataset = DATASET_LABELS.get(metadata.get("dataset", ""), "dataset")
        group_by = metadata.get("group_by")

        if group_by:
            group_label = COLUMN_LABELS.get(group_by, group_by.replace("_", " ").title())
            return f"{dataset} grouped by {group_label}. Found {row_count} groups."
        else:
            if row_count == 1 and "count" in df.columns:
                count_val = df["count"].iloc[0]
                return f"Total {dataset}: {count_val:,} records"
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
    "crime_2022": {
        "name": "Crime Data 2022 (Part 1 Offenses)",
        "source": "Syracuse Open Data Portal",
        "url": "https://data.syr.gov/datasets/crime-data-2022",
        "date_range": "January - December 2022",
        "update_frequency": "Static (annual)",
        "caveats": [
            "Only Part 1 offenses (serious crimes) included",
            "Addresses generalized to block level for privacy",
            "Reflects reported crimes, not all incidents",
            "Under-reporting varies by crime type",
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

    result = run_query(req.question)

    if result.get("error"):
        return QueryResponse(
            success=False,
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
        count_cols = ["count", "primary_count", "secondary_count"]
        category_col = None
        value_cols = []

        for col in raw_columns:
            if col.lower() in count_cols:
                value_cols.append(col)
            elif category_col is None:
                category_col = col

        if category_col and value_cols:
            chart_df = df.head(15)

            if metadata.get("query_type") == "join" and metadata.get("join_type") in ("zip", "neighborhood") and len(value_cols) >= 2:
                # Grouped bar chart data
                chart_data = {
                    "type": "grouped_bar",
                    "labels": chart_df[category_col].tolist(),
                    "datasets": [
                        {
                            "label": DATASET_LABELS.get(metadata.get("primary_dataset", ""), "Primary"),
                            "data": chart_df[value_cols[0]].tolist(),
                        },
                        {
                            "label": DATASET_LABELS.get(metadata.get("secondary_dataset", ""), "Secondary"),
                            "data": chart_df[value_cols[1]].tolist() if len(value_cols) > 1 else [],
                        }
                    ],
                    "x_label": get_readable_column(category_col, metadata),
                }
            else:
                # Simple bar chart data
                chart_data = {
                    "type": "bar",
                    "labels": chart_df[category_col].tolist(),
                    "values": chart_df[value_cols[0]].tolist(),
                    "x_label": get_readable_column(category_col, metadata),
                    "y_label": get_readable_column(value_cols[0], metadata),
                }

    # Generate description
    description = generate_description(df, metadata)

    # Generate insights
    insights = generate_insights(df, metadata, req.question)

    # Get citations based on datasets used
    citations = []
    if metadata.get("query_type") == "join":
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
