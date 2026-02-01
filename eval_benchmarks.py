"""
Evaluation benchmarks for Ask Syracuse Data.
Tests natural language query parsing accuracy and result correctness.

Run: python eval_benchmarks.py
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from main import run_query
from llm.intent_parser import parse_intent, _heuristic_intent


@dataclass
class TestCase:
    """A single test case for query evaluation."""
    name: str
    question: str
    expected_dataset: str
    expected_query_type: str = "single"
    expected_group_by: Optional[str] = None
    expected_join_type: Optional[str] = None
    expected_primary: Optional[str] = None
    expected_secondary: Optional[str] = None
    min_result_count: int = 0
    max_result_count: Optional[int] = None
    expected_columns: Optional[List[str]] = None


# =============================================================================
# BENCHMARK TEST CASES
# =============================================================================
BENCHMARK_TESTS: List[TestCase] = [
    # Single-dataset queries - Code Violations
    TestCase(
        name="total_violations",
        question="How many code violations are there?",
        expected_dataset="violations",
        expected_query_type="single",
        expected_group_by=None,
        min_result_count=1,
        max_result_count=1,
        expected_columns=["count"],
    ),
    TestCase(
        name="violations_by_neighborhood",
        question="Violations by neighborhood",
        expected_dataset="violations",
        expected_query_type="single",
        expected_group_by="neighborhood",
        min_result_count=5,
    ),
    TestCase(
        name="violations_by_zip",
        question="How many violations by zip code?",
        expected_dataset="violations",
        expected_query_type="single",
        expected_group_by="complaint_zip",
        min_result_count=5,
    ),
    TestCase(
        name="violations_by_status",
        question="Show violations by status",
        expected_dataset="violations",
        expected_query_type="single",
        expected_group_by="status_type_name",
        min_result_count=2,
    ),

    # Single-dataset queries - Crime
    TestCase(
        name="crime_by_type",
        question="What types of crimes occurred in 2022?",
        expected_dataset="crime_2022",
        expected_query_type="single",
        expected_group_by="code_defined",
        min_result_count=3,
    ),
    TestCase(
        name="crime_by_neighborhood",
        question="Crime counts by neighborhood",
        expected_dataset="crime_2022",
        expected_query_type="single",
        expected_group_by="neighborhood",
        min_result_count=5,
    ),

    # Single-dataset queries - Vacant Properties
    TestCase(
        name="vacant_by_neighborhood",
        question="Vacant properties by neighborhood",
        expected_dataset="vacant_properties",
        expected_query_type="single",
        expected_group_by="neighborhood",
        min_result_count=5,
    ),
    TestCase(
        name="total_vacant",
        question="How many vacant properties are there?",
        expected_dataset="vacant_properties",
        expected_query_type="single",
        expected_group_by=None,
        min_result_count=1,
        max_result_count=1,
    ),

    # Single-dataset queries - Rental Registry
    TestCase(
        name="rentals_by_zip",
        question="Rental properties by zip code",
        expected_dataset="rental_registry",
        expected_query_type="single",
        expected_group_by="zip",
        min_result_count=5,
    ),

    # Cross-dataset join queries
    TestCase(
        name="rental_violations_zip",
        question="Which zip codes have rental properties with violations?",
        expected_dataset="rental_registry",
        expected_query_type="join",
        expected_join_type="zip",
        expected_primary="rental_registry",
        expected_secondary="violations",
        min_result_count=5,
    ),
    TestCase(
        name="rental_violations_sbl",
        question="Which specific rental properties have code violations?",
        expected_dataset="rental_registry",
        expected_query_type="join",
        expected_join_type="sbl",
        expected_primary="rental_registry",
        expected_secondary="violations",
        min_result_count=1,
    ),
    TestCase(
        name="vacant_violations",
        question="How many violations are there in vacant properties?",
        expected_dataset="vacant_properties",
        expected_query_type="join",
        expected_join_type="sbl",
        expected_primary="vacant_properties",
        expected_secondary="violations",
        min_result_count=1,
    ),
]


@dataclass
class TestResult:
    """Result of a single test case execution."""
    name: str
    passed: bool
    errors: List[str]
    warnings: List[str]
    intent: Optional[Dict[str, Any]] = None
    result_count: int = 0
    execution_time_ms: float = 0


def evaluate_intent(test: TestCase, intent: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Evaluate if parsed intent matches expected values."""
    errors = []

    query_type = intent.get("query_type", "single")
    if query_type != test.expected_query_type:
        errors.append(f"query_type: expected '{test.expected_query_type}', got '{query_type}'")

    if test.expected_query_type == "single":
        dataset = intent.get("dataset", "")
        if dataset != test.expected_dataset:
            errors.append(f"dataset: expected '{test.expected_dataset}', got '{dataset}'")

        if test.expected_group_by is not None:
            group_by = intent.get("group_by")
            if group_by != test.expected_group_by:
                errors.append(f"group_by: expected '{test.expected_group_by}', got '{group_by}'")

    elif test.expected_query_type == "join":
        if test.expected_primary:
            primary = intent.get("primary_dataset", "")
            if primary != test.expected_primary:
                errors.append(f"primary_dataset: expected '{test.expected_primary}', got '{primary}'")

        if test.expected_secondary:
            secondary = intent.get("secondary_dataset", "")
            if secondary != test.expected_secondary:
                errors.append(f"secondary_dataset: expected '{test.expected_secondary}', got '{secondary}'")

        if test.expected_join_type:
            join_type = intent.get("join_type", "")
            if join_type != test.expected_join_type:
                errors.append(f"join_type: expected '{test.expected_join_type}', got '{join_type}'")

    return len(errors) == 0, errors


def evaluate_result(test: TestCase, result: Dict[str, Any]) -> tuple[bool, List[str], List[str]]:
    """Evaluate if query result matches expected values."""
    errors = []
    warnings = []

    if result.get("error"):
        errors.append(f"Query error: {result['error']}")
        return False, errors, warnings

    df = result.get("result")
    if df is None or not isinstance(df, pd.DataFrame):
        errors.append("No result dataframe returned")
        return False, errors, warnings

    row_count = len(df)

    if row_count < test.min_result_count:
        errors.append(f"Result count {row_count} < minimum expected {test.min_result_count}")

    if test.max_result_count is not None and row_count > test.max_result_count:
        errors.append(f"Result count {row_count} > maximum expected {test.max_result_count}")

    if test.expected_columns:
        actual_cols = set(df.columns)
        expected_cols = set(test.expected_columns)
        missing = expected_cols - actual_cols
        if missing:
            errors.append(f"Missing expected columns: {missing}")

    # Check validation results
    validation = result.get("validation", {})
    if validation:
        if not validation.get("passed", True):
            warnings.append("Validation failed: " + "; ".join(validation.get("errors", [])))
        if validation.get("warnings"):
            warnings.extend(validation.get("warnings", []))

    return len(errors) == 0, errors, warnings


def run_test(test: TestCase, use_llm: bool = False) -> TestResult:
    """Run a single test case."""
    import time

    errors = []
    warnings = []

    start = time.time()

    # Test intent parsing (heuristic only for consistency)
    try:
        if use_llm:
            # Use full pipeline
            response = run_query(test.question)
            intent = response.get("intent", {})
        else:
            # Test heuristic parsing only
            intent = _heuristic_intent(test.question)
            if intent is None:
                errors.append("Heuristic parsing returned None")
                return TestResult(
                    name=test.name,
                    passed=False,
                    errors=errors,
                    warnings=warnings,
                    intent=None,
                )

        intent_passed, intent_errors = evaluate_intent(test, intent)
        if not intent_passed:
            errors.extend(intent_errors)

    except Exception as e:
        errors.append(f"Intent parsing error: {str(e)}")
        return TestResult(
            name=test.name,
            passed=False,
            errors=errors,
            warnings=warnings,
        )

    # Test full query execution
    try:
        response = run_query(test.question)
        result_passed, result_errors, result_warnings = evaluate_result(test, response)

        if not result_passed:
            errors.extend(result_errors)
        warnings.extend(result_warnings)

        result_count = len(response.get("result", [])) if isinstance(response.get("result"), pd.DataFrame) else 0

    except Exception as e:
        errors.append(f"Query execution error: {str(e)}")
        result_count = 0

    elapsed = (time.time() - start) * 1000

    return TestResult(
        name=test.name,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        intent=intent,
        result_count=result_count,
        execution_time_ms=elapsed,
    )


def run_all_benchmarks(use_llm: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Run all benchmark tests and return summary."""
    results = []

    for test in BENCHMARK_TESTS:
        if verbose:
            print(f"Running: {test.name}...", end=" ")

        result = run_test(test, use_llm=use_llm)
        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} ({result.execution_time_ms:.0f}ms)")
            if not result.passed:
                for err in result.errors:
                    print(f"  - {err}")

    # Calculate summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_time = sum(r.execution_time_ms for r in results)

    summary = {
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(results) * 100 if results else 0,
        "total_time_ms": total_time,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "errors": r.errors,
                "warnings": r.warnings,
                "result_count": r.result_count,
                "time_ms": r.execution_time_ms,
            }
            for r in results
        ],
    }

    return summary


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass Rate: {summary['pass_rate']:.1f}%")
    print(f"Total Time: {summary['total_time_ms']:.0f}ms")

    if summary['failed'] > 0:
        print("\nFailed Tests:")
        for r in summary['results']:
            if not r['passed']:
                print(f"  - {r['name']}")
                for err in r['errors']:
                    print(f"      {err}")

    print("=" * 60)


def main():
    """Run benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Ask Syracuse Data evaluation benchmarks")
    parser.add_argument("--llm", action="store_true", help="Use LLM for intent parsing")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    print("Ask Syracuse Data - Evaluation Benchmarks")
    print("-" * 40)

    summary = run_all_benchmarks(use_llm=args.llm, verbose=not args.quiet)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)

    # Exit with error code if any tests failed
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
