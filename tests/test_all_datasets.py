"""Systematic test of all 16 datasets with multiple query types."""
import requests
import time
import sys

URL = "http://127.0.0.1:8000/api/query"

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


def run_tests():
    results = []
    current_dataset = None

    for dataset, question, qtype in tests:
        if dataset != current_dataset:
            current_dataset = dataset
            print(f"\n===== {dataset.upper()} =====")

        start = time.time()
        try:
            r = requests.post(URL, json={"question": question}, timeout=30)
            data = r.json()
            elapsed = int((time.time() - start) * 1000)
            success = data.get("success", False)
            rows = len(data.get("data", []))
            chart_data = data.get("chart_data")
            chart = chart_data.get("type", "none") if chart_data else "none"
            cols = data.get("columns", [])

            if success:
                status = "PASS"
                print(f"  PASS ({elapsed}ms) | {question}")
                # Show first 2 data rows
                for row in data.get("data", [])[:2]:
                    vals = [f"{k}={v}" for k, v in row.items()]
                    print(f"    -> {', '.join(vals)}")
                print(f"    [{rows} rows, chart={chart}]")
            else:
                status = "FAIL"
                err = data.get("error", "unknown error")
                print(f"  FAIL ({elapsed}ms) | {question}")
                print(f"    ERROR: {err}")

            results.append((dataset, question, status, elapsed))
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            print(f"  ERROR ({elapsed}ms) | {question}")
            print(f"    {e}")
            results.append((dataset, question, "ERROR", elapsed))

    # Summary
    print(f"\n{'=' * 60}")
    print("DATASET TEST SUMMARY")
    print(f"{'=' * 60}")
    passed = sum(1 for _, _, s, _ in results if s == "PASS")
    failed = sum(1 for _, _, s, _ in results if s != "PASS")
    total_time = sum(t for _, _, _, t in results)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"Total Time: {total_time}ms")
    if failed:
        print(f"\nFAILED TESTS:")
        for ds, q, s, t in results:
            if s != "PASS":
                print(f"  [{ds}] {q} -> {s}")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
