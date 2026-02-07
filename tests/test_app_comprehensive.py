"""
Comprehensive test suite for Ask Syracuse Data.
Tests 30 questions ranging from easy to hard.
"""
from __future__ import annotations
import sys
import pandas as pd
from pipeline.main import run_query

# =============================================================================
# TEST QUESTIONS (Easy -> Medium -> Hard)
# =============================================================================

TEST_QUESTIONS = [
    # === EASY: Basic counts ===
    {
        "id": 1,
        "difficulty": "easy",
        "question": "How many code violations are there?",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and "count" in r["result"].columns,
    },
    {
        "id": 2,
        "difficulty": "easy",
        "question": "How many vacant properties are there?",
        "expected_dataset": "vacant_properties",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and len(r["result"]) == 1,
    },
    {
        "id": 3,
        "difficulty": "easy",
        "question": "How many rental properties are registered?",
        "expected_dataset": "rental_registry",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 4,
        "difficulty": "easy",
        "question": "How many crimes were reported in 2022?",
        "expected_dataset": "crime_2022",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 5,
        "difficulty": "easy",
        "question": "Total violations",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },

    # === EASY-MEDIUM: Grouped counts ===
    {
        "id": 6,
        "difficulty": "easy-medium",
        "question": "Violations by neighborhood",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and len(r["result"]) > 5,
    },
    {
        "id": 7,
        "difficulty": "easy-medium",
        "question": "Code violations by zip code",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and "complaint_zip" in r["result"].columns,
    },
    {
        "id": 8,
        "difficulty": "easy-medium",
        "question": "Show violations by status",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and "status_type_name" in r["result"].columns,
    },
    {
        "id": 9,
        "difficulty": "easy-medium",
        "question": "What types of crimes occurred?",
        "expected_dataset": "crime_2022",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and "code_defined" in r["result"].columns,
    },
    {
        "id": 10,
        "difficulty": "easy-medium",
        "question": "Vacant properties by neighborhood",
        "expected_dataset": "vacant_properties",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and len(r["result"]) > 3,
    },
    {
        "id": 11,
        "difficulty": "easy-medium",
        "question": "Crime counts by neighborhood",
        "expected_dataset": "crime_2022",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and "neighborhood" in r["result"].columns,
    },
    {
        "id": 12,
        "difficulty": "easy-medium",
        "question": "Rental properties by zip code",
        "expected_dataset": "rental_registry",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and "zip" in r["result"].columns,
    },

    # === MEDIUM: Specific groupings ===
    {
        "id": 13,
        "difficulty": "medium",
        "question": "Which neighborhoods have the most violations?",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and len(r["result"]) > 5,
    },
    {
        "id": 14,
        "difficulty": "medium",
        "question": "What are the most common crime types?",
        "expected_dataset": "crime_2022",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and "code_defined" in r["result"].columns,
    },
    {
        "id": 15,
        "difficulty": "medium",
        "question": "How many violations are open vs closed?",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 16,
        "difficulty": "medium",
        "question": "Vacant properties by zip",
        "expected_dataset": "vacant_properties",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 17,
        "difficulty": "medium",
        "question": "Show crime statistics by neighborhood",
        "expected_dataset": "crime_2022",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 18,
        "difficulty": "medium",
        "question": "How many arrests were made?",
        "expected_dataset": "crime_2022",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },

    # === MEDIUM-HARD: Cross-dataset joins ===
    {
        "id": 19,
        "difficulty": "medium-hard",
        "question": "Which zip codes have rental properties with violations?",
        "expected_dataset": "rental_registry",
        "expected_type": "join",
        "check": lambda r: r.get("metadata", {}).get("query_type") == "join",
    },
    {
        "id": 20,
        "difficulty": "medium-hard",
        "question": "How many violations are there in vacant properties?",
        "expected_dataset": "vacant_properties",
        "expected_type": "join",
        "check": lambda r: r.get("metadata", {}).get("query_type") == "join",
    },
    {
        "id": 21,
        "difficulty": "medium-hard",
        "question": "Which specific rental properties have code violations?",
        "expected_dataset": "rental_registry",
        "expected_type": "join",
        "check": lambda r: r.get("metadata", {}).get("query_type") == "join" and r.get("metadata", {}).get("join_type") == "sbl",
    },
    {
        "id": 22,
        "difficulty": "medium-hard",
        "question": "Violations in areas with vacant properties",
        "expected_dataset": "vacant_properties",
        "expected_type": "join",
        "check": lambda r: r.get("metadata", {}).get("query_type") == "join",
    },
    {
        "id": 23,
        "difficulty": "medium-hard",
        "question": "Rental properties with violations by zip",
        "expected_dataset": "rental_registry",
        "expected_type": "join",
        "check": lambda r: r.get("result") is not None,
    },

    # === HARD: Complex queries ===
    {
        "id": 24,
        "difficulty": "hard",
        "question": "Which neighborhoods have both vacant properties and code violations?",
        "expected_dataset": "vacant_properties",
        "expected_type": "join",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 25,
        "difficulty": "hard",
        "question": "Compare rental properties and violations by zip code",
        "expected_dataset": "rental_registry",
        "expected_type": "join",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 26,
        "difficulty": "hard",
        "question": "How many vacant properties have active violations?",
        "expected_dataset": "vacant_properties",
        "expected_type": "join",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 27,
        "difficulty": "hard",
        "question": "Show me violation hotspots by neighborhood",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None and len(r["result"]) > 5,
    },
    {
        "id": 28,
        "difficulty": "hard",
        "question": "Which rental properties are also marked as vacant?",
        "expected_dataset": "rental_registry",
        "expected_type": "join",
        "check": lambda r: r.get("metadata", {}).get("query_type") == "join",
    },
    {
        "id": 29,
        "difficulty": "hard",
        "question": "Properties with the most code violations",
        "expected_dataset": "violations",
        "expected_type": "single",
        "check": lambda r: r.get("result") is not None,
    },
    {
        "id": 30,
        "difficulty": "hard",
        "question": "Analyze violations across rental and vacant properties",
        "expected_dataset": "rental_registry",
        "expected_type": "join",
        "check": lambda r: r.get("result") is not None,
    },
]


def run_test(test: dict) -> dict:
    """Run a single test and return results."""
    question = test["question"]

    try:
        result = run_query(question)

        # Check for errors
        if result.get("error"):
            return {
                "id": test["id"],
                "question": question,
                "difficulty": test["difficulty"],
                "passed": False,
                "error": result["error"],
                "details": None,
            }

        # Check if result matches expectations
        custom_check_passed = test["check"](result) if test.get("check") else True

        # Get result details
        df = result.get("result")
        row_count = len(df) if isinstance(df, pd.DataFrame) else 0

        metadata = result.get("metadata", {})
        query_type = metadata.get("query_type", "single")
        dataset = metadata.get("dataset") or metadata.get("primary_dataset", "unknown")

        # Validation info
        validation = result.get("validation", {})
        validation_passed = validation.get("passed", True) if validation else True
        validation_warnings = validation.get("warnings", []) if validation else []

        # Bias warnings
        bias_warnings = result.get("bias_warnings", [])

        passed = custom_check_passed and result.get("result") is not None

        return {
            "id": test["id"],
            "question": question,
            "difficulty": test["difficulty"],
            "passed": passed,
            "error": None,
            "details": {
                "query_type": query_type,
                "dataset": dataset,
                "row_count": row_count,
                "validation_passed": validation_passed,
                "validation_warnings": len(validation_warnings),
                "bias_warnings": len(bias_warnings),
                "sql": result.get("sql", "")[:100] + "..." if result.get("sql") else None,
            },
        }

    except Exception as e:
        return {
            "id": test["id"],
            "question": question,
            "difficulty": test["difficulty"],
            "passed": False,
            "error": str(e),
            "details": None,
        }


def run_all_tests(verbose: bool = True):
    """Run all tests and print results."""
    results = []

    print("=" * 80)
    print("ASK SYRACUSE DATA - COMPREHENSIVE TEST SUITE (30 Questions)")
    print("=" * 80)
    print()

    for test in TEST_QUESTIONS:
        if verbose:
            print(f"[{test['id']:2d}] ({test['difficulty']:12s}) {test['question'][:50]}...", end=" ")

        result = run_test(test)
        results.append(result)

        if verbose:
            if result["passed"]:
                details = result["details"]
                print(f"PASS ({details['query_type']}, {details['row_count']} rows)")
            else:
                print(f"FAIL - {result['error']}")

    # Summary by difficulty
    print()
    print("=" * 80)
    print("RESULTS BY DIFFICULTY")
    print("=" * 80)

    difficulties = ["easy", "easy-medium", "medium", "medium-hard", "hard"]
    for diff in difficulties:
        diff_results = [r for r in results if r["difficulty"] == diff]
        passed = sum(1 for r in diff_results if r["passed"])
        total = len(diff_results)
        pct = (passed / total * 100) if total > 0 else 0
        status = "OK" if passed == total else "ISSUES"
        print(f"  {diff:12s}: {passed}/{total} passed ({pct:.0f}%) [{status}]")

    # Overall summary
    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_passed = sum(1 for r in results if r["passed"])
    total_failed = len(results) - total_passed

    print(f"  Total Tests: {len(results)}")
    print(f"  Passed:      {total_passed}")
    print(f"  Failed:      {total_failed}")
    print(f"  Pass Rate:   {total_passed/len(results)*100:.1f}%")

    # Show failed tests
    if total_failed > 0:
        print()
        print("FAILED TESTS:")
        print("-" * 40)
        for r in results:
            if not r["passed"]:
                print(f"  [{r['id']}] {r['question']}")
                print(f"      Error: {r['error']}")

    # Show sample results with data
    print()
    print("=" * 80)
    print("SAMPLE QUERY RESULTS")
    print("=" * 80)

    # Show first few successful results
    sample_ids = [1, 6, 11, 19, 20]
    for test_id in sample_ids:
        test = next((t for t in TEST_QUESTIONS if t["id"] == test_id), None)
        if test:
            print(f"\n[{test_id}] {test['question']}")
            result = run_query(test["question"])
            if result.get("result") is not None:
                df = result["result"]
                print(f"    Dataset: {result.get('metadata', {}).get('dataset') or result.get('metadata', {}).get('primary_dataset')}")
                print(f"    Rows: {len(df)}")
                print(f"    SQL: {result.get('sql', 'N/A')[:80]}...")
                print("    Data Preview:")
                print(df.head(5).to_string(index=False).replace('\n', '\n    '))

    print()
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_all_tests(verbose=True)

    # Exit with error code if any tests failed
    failed = sum(1 for r in results if not r["passed"])
    sys.exit(0 if failed == 0 else 1)
