"""Systematic test of all 16 datasets with multiple query types.

This suite runs against the FastAPI app via TestClient (no separate server needed).
If the local `data/raw/` snapshots are not present (e.g., in CI), the tests are skipped.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DISABLE_RATE_LIMITS", "1")
if os.getenv("RUN_DATASET_API_TESTS") != "1":
    pytest.skip("Set RUN_DATASET_API_TESTS=1 to run dataset API tests", allow_module_level=True)

tests = [
    # 1. VIOLATIONS
    ("violations", "How many violations are there?", "total"),
    ("violations", "Violations by neighborhood", "grouped"),
    ("violations", "Violations by year", "temporal"),
    # 2. VACANT PROPERTIES
    ("vacant_properties", "How many vacant properties are there?", "total"),
    ("vacant_properties", "Vacant properties by neighborhood", "grouped"),
    # 3. CRIME
    ("crime", "How many crimes are there?", "total"),
    ("crime", "Crime by year", "temporal"),
    ("crime", "Crime by neighborhood", "grouped"),
    # 4. RENTAL REGISTRY
    ("rental_registry", "How many rental properties are there?", "total"),
    ("rental_registry", "Rental properties by zip code", "grouped"),
    # 5. UNFIT PROPERTIES
    ("unfit_properties", "How many unfit properties are there?", "total"),
    ("unfit_properties", "Unfit properties by zip code", "grouped"),
    ("unfit_properties", "Unfit properties by department", "grouped"),
    # 6. TRASH PICKUP
    ("trash_pickup", "How many properties have trash pickup on Monday?", "filtered"),
    ("trash_pickup", "Trash pickup by collection day", "grouped"),
    # 7. HISTORICAL PROPERTIES
    ("historical_properties", "How many historical properties are there?", "total"),
    ("historical_properties", "Historical properties by NR eligibility", "grouped"),
    ("historical_properties", "Historical properties by zip code", "grouped"),
    # 8. ASSESSMENT ROLL
    ("assessment_roll", "How many properties by class description?", "grouped"),
    ("assessment_roll", "What is the average property assessment?", "metric"),
    ("assessment_roll", "Average property assessment by zip code", "grouped_metric"),
    # 9. CITYLINE REQUESTS
    ("cityline_requests", "What are the most common cityline complaints?", "grouped"),
    ("cityline_requests", "Service requests by agency", "grouped"),
    ("cityline_requests", "Service requests by year", "temporal"),
    # 10. SNOW ROUTES
    ("snow_routes", "How many snow routes by zip code?", "grouped"),
    # 11. BIKE SUITABILITY
    ("bike_suitability", "Bike suitability by rating", "grouped"),
    # 12. BIKE INFRASTRUCTURE
    ("bike_infrastructure", "Bike infrastructure by type", "grouped"),
    ("bike_infrastructure", "How many miles of bike lanes are there?", "metric"),
    # 13. PARKING VIOLATIONS
    ("parking_violations", "How many parking violations by type?", "grouped"),
    ("parking_violations", "Parking violations by year", "temporal"),
    ("parking_violations", "Average parking fine by type", "grouped_metric"),
    # 14. PERMIT REQUESTS
    ("permit_requests", "How many building permits by type?", "grouped"),
    ("permit_requests", "Permits by year", "temporal"),
    # 15. TREE INVENTORY
    ("tree_inventory", "Most common tree species in Syracuse", "grouped"),
    ("tree_inventory", "How many trees by neighborhood?", "grouped"),
    # 16. LEAD TESTING
    ("lead_testing", "Lead testing results by census tract", "grouped"),
    ("lead_testing", "Lead testing by year", "temporal"),
]

@pytest.fixture(scope="session")
def client() -> TestClient:
    data_dir = Path("data") / "raw"
    if not data_dir.exists():
        pytest.skip("data/raw not present; skipping dataset API tests")

    from app import app

    return TestClient(app)


@pytest.mark.parametrize("dataset,question,qtype", tests)
def test_api_query_succeeds(client: TestClient, dataset: str, question: str, qtype: str):
    resp = client.post("/api/query", json={"question": question})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("success") is True, body.get("error") or body
    assert isinstance(body.get("data"), list)
    assert isinstance(body.get("columns"), list)
