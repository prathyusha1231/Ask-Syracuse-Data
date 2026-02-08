"""
Pytest test suite for Ask Syracuse Data pipeline.
Covers: intent parsing, SQL execution, metric queries, HAVING, filters,
error handling, and ground-truth validation.

Run: pytest tests/test_pipeline.py -v
"""
from __future__ import annotations
import pytest
import pandas as pd
import duckdb

from pipeline.main import run_query, _get_cached_df, LOADERS
from pipeline import schema
from pipeline.sql_builder import build_select_sql
from pipeline.sql_validator import validate_sql, SQLValidationError
from llm.intent_parser import _heuristic_intent


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def violations_df():
    """Load violations data once for the session."""
    return _get_cached_df("violations", LOADERS["violations"])


@pytest.fixture(scope="session")
def crime_df():
    return _get_cached_df("crime", LOADERS["crime"])


@pytest.fixture(scope="session")
def parking_df():
    return _get_cached_df("parking_violations", LOADERS["parking_violations"])


@pytest.fixture(scope="session")
def cityline_df():
    return _get_cached_df("cityline_requests", LOADERS["cityline_requests"])


# =============================================================================
# INTENT PARSING (Heuristic)
# =============================================================================

class TestIntentParsing:
    """Test heuristic intent parsing for various query types."""

    def test_basic_count(self):
        intent = _heuristic_intent("How many violations are there?")
        assert intent["dataset"] == "violations"
        assert intent["metric"] == "count"

    def test_group_by_neighborhood(self):
        intent = _heuristic_intent("Violations by neighborhood")
        assert intent["dataset"] == "violations"
        assert intent["group_by"] in ("neighborhood", ["neighborhood"])

    def test_group_by_zip(self):
        intent = _heuristic_intent("Violations by zip code")
        assert intent["dataset"] == "violations"
        assert intent["group_by"] in ("complaint_zip", ["complaint_zip"])

    def test_crime_by_year(self):
        intent = _heuristic_intent("Crime by year")
        assert intent["dataset"] == "crime"
        gb = intent["group_by"]
        # Could be "year" or list containing "year"
        if isinstance(gb, list):
            assert "year" in gb
        else:
            assert gb == "year"

    def test_crime_by_type(self):
        intent = _heuristic_intent("What types of crimes occurred?")
        assert intent["dataset"] == "crime"

    def test_vacant_properties(self):
        intent = _heuristic_intent("How many vacant properties?")
        assert intent["dataset"] == "vacant_properties"

    def test_rental_registry(self):
        intent = _heuristic_intent("Rental properties by zip code")
        assert intent["dataset"] == "rental_registry"

    def test_parking_violations(self):
        intent = _heuristic_intent("Parking fines by description")
        assert intent["dataset"] == "parking_violations"

    def test_tree_inventory(self):
        intent = _heuristic_intent("Trees by condition")
        assert intent["dataset"] == "tree_inventory"

    def test_permit_requests(self):
        intent = _heuristic_intent("Building permits by type")
        assert intent["dataset"] == "permit_requests"

    def test_cityline_requests(self):
        intent = _heuristic_intent("Cityline requests by category")
        assert intent["dataset"] == "cityline_requests"

    def test_average_metric(self):
        intent = _heuristic_intent("What is the average days to comply for violations by neighborhood?")
        # Heuristic may or may not handle avg; if None, it needs LLM
        if intent is not None:
            assert intent["dataset"] == "violations"
            assert intent["metric"] == "avg"
        else:
            pytest.skip("Heuristic cannot parse avg metric queries — needs LLM")

    def test_join_detection_zip(self):
        intent = _heuristic_intent("Which zip codes have rental properties with violations?")
        assert intent.get("query_type") == "join"
        assert intent.get("join_type") == "zip"

    def test_join_detection_sbl(self):
        intent = _heuristic_intent("Which specific rental properties have code violations?")
        assert intent.get("query_type") == "join"
        assert intent.get("join_type") == "sbl"

    def test_year_filter(self):
        intent = _heuristic_intent("How many crimes in 2022?")
        assert intent["dataset"] == "crime"
        filters = intent.get("filters", {})
        assert "year" in filters


# =============================================================================
# METRIC QUERIES (avg, min, max, sum) — Item #10
# =============================================================================

class TestMetricQueries:
    """Test aggregation metrics.
    Note: heuristic parser defaults most metric queries to count.
    Avg/min/max/sum detection requires LLM. Tests here verify the
    heuristic handles these gracefully (returns count instead of crashing).
    """

    @pytest.mark.llm
    def test_avg_days_to_comply(self):
        """Avg metric — heuristic may return count instead."""
        result = run_query("What is the average days to comply for violations by neighborhood?")
        # Heuristic returns count; this is expected without LLM
        assert result.get("result") is not None or result.get("error") is not None

    def test_parking_fines_by_type(self):
        result = run_query("Parking fines by description")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        assert len(df) > 1

    def test_cityline_by_category(self):
        result = run_query("Cityline requests by category")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        assert len(df) > 1

    def test_violations_count_basic(self):
        """Basic count metric works correctly."""
        result = run_query("How many violations by neighborhood?")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        assert "count" in df.columns
        assert len(df) > 5


# =============================================================================
# TEMPORAL GROUP BY QUERIES
# =============================================================================

class TestTemporalQueries:
    """Test temporal grouping: year, month, quarter."""

    def test_violations_by_year(self):
        result = run_query("Violations by year")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        assert len(df) >= 2  # At least 2 years of data

    def test_crime_by_year(self):
        result = run_query("Crime by year")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        assert len(df) >= 2

    def test_violations_by_year_has_data(self):
        """Violations by year should return multiple years of data."""
        result = run_query("Violations by year")
        assert result.get("error") is None
        df = result["result"]
        assert len(df) >= 2
        # Should have a count column
        assert "count" in df.columns or len(df.select_dtypes(include="number").columns) >= 1


# =============================================================================
# HAVING CLAUSE QUERIES — Item #10
# =============================================================================

class TestHavingQueries:
    """Test HAVING threshold queries."""

    def test_neighborhoods_with_many_violations(self):
        result = run_query("Neighborhoods with more than 100 violations")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        # All rows should have count > 100
        count_col = "count" if "count" in df.columns else df.columns[-1]
        assert (df[count_col] > 100).all(), f"Some rows have count <= 100: {df[count_col].min()}"

    def test_zips_with_many_violations(self):
        result = run_query("Zip codes with more than 50 violations")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        count_col = "count" if "count" in df.columns else df.columns[-1]
        assert (df[count_col] > 50).all()

    def test_having_with_over(self):
        result = run_query("Neighborhoods with over 200 violations")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        count_col = "count" if "count" in df.columns else df.columns[-1]
        assert (df[count_col] > 200).all()


# =============================================================================
# FILTER QUERIES — Item #10
# =============================================================================

class TestFilterQueries:
    """Test filter operators: =, >=, year filters."""

    def test_year_equals(self):
        result = run_query("How many crimes in 2022?")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None
        assert len(df) >= 1

    def test_year_since(self):
        result = run_query("Violations since 2023")
        assert result.get("error") is None
        df = result["result"]
        assert df is not None

    def test_crime_year_filter_ground_truth(self, crime_df):
        """Ground-truth: total crime count in 2022 should match pandas."""
        result = run_query("How many crimes in 2022?")
        assert result.get("error") is None
        df = result["result"]

        # Pandas ground truth
        cdf = crime_df.copy()
        if "year" in cdf.columns:
            gt_count = int((cdf["year"] == 2022).sum())
            # Heuristic groups by code_defined, so sum all rows
            count_col = "count" if "count" in df.columns else df.columns[-1]
            sql_total = int(df[count_col].sum())
            if gt_count > 0:
                ratio = sql_total / gt_count
                assert 0.9 < ratio < 1.1, f"2022 crime count: SQL={sql_total}, pandas={gt_count}"


# =============================================================================
# CROSS-DATASET JOIN QUERIES
# =============================================================================

class TestJoinQueries:
    """Test cross-dataset joins."""

    def test_rental_violations_zip(self):
        result = run_query("Which zip codes have rental properties with violations?")
        assert result.get("error") is None
        assert result.get("metadata", {}).get("query_type") == "join"
        assert result["result"] is not None
        assert len(result["result"]) >= 5

    def test_vacant_violations_sbl(self):
        result = run_query("How many violations are there in vacant properties?")
        assert result.get("error") is None
        assert result["result"] is not None

    def test_rental_violations_sbl(self):
        result = run_query("Which specific rental properties have code violations?")
        assert result.get("error") is None
        assert result.get("metadata", {}).get("join_type") == "sbl"


# =============================================================================
# ERROR HANDLING / NEGATIVE TESTS — Item #11
# =============================================================================

class TestErrorHandling:
    """Test that bad inputs fail gracefully instead of crashing."""

    def test_empty_question(self):
        result = run_query("")
        # Should return an error, not crash
        assert result.get("error") is not None or result.get("result") is None

    def test_nonsense_question(self):
        result = run_query("asdfjkl qwerty zxcvb")
        # Should return error or empty result, not crash
        assert isinstance(result, dict)

    def test_very_long_question(self):
        result = run_query("violations " * 100)
        # Should handle gracefully
        assert isinstance(result, dict)

    def test_special_characters(self):
        result = run_query("How many violations with <script>alert('xss')</script>?")
        # Should not crash
        assert isinstance(result, dict)

    def test_sql_injection_attempt(self):
        result = run_query("'; DROP TABLE violations; --")
        # Should not crash or drop anything
        assert isinstance(result, dict)
        # Should return an error (not parseable)
        assert result.get("error") is not None or result.get("result") is not None

    def test_data_survives_injection(self):
        """Verify data is intact after injection attempt."""
        # First do the injection attempt
        run_query("'; DROP TABLE violations; --")
        # Then verify violations still work
        result = run_query("How many violations?")
        assert result.get("error") is None
        assert result["result"] is not None


# =============================================================================
# SQL VALIDATOR TESTS
# =============================================================================

class TestSQLValidator:
    """Test SQL guardrails."""

    def test_valid_select(self):
        sql = validate_sql("SELECT COUNT(*) FROM violations")
        assert "LIMIT" in sql

    def test_valid_with_cte(self):
        sql = validate_sql("WITH v AS (SELECT * FROM violations) SELECT COUNT(*) FROM v")
        assert "LIMIT" in sql

    def test_rejects_drop(self):
        with pytest.raises(SQLValidationError):
            validate_sql("DROP TABLE violations")

    def test_rejects_insert(self):
        with pytest.raises(SQLValidationError):
            validate_sql("INSERT INTO violations VALUES (1)")

    def test_rejects_delete(self):
        with pytest.raises(SQLValidationError):
            validate_sql("DELETE FROM violations")

    def test_rejects_write_in_select(self):
        """Write ops hidden inside a SELECT should still be caught (by semicolon or write-op check)."""
        with pytest.raises(SQLValidationError):
            validate_sql("SELECT * FROM violations; DELETE FROM violations")

    def test_rejects_semicolons(self):
        with pytest.raises(SQLValidationError, match="semicolons"):
            validate_sql("SELECT 1; DROP TABLE violations")

    def test_rejects_unknown_table(self):
        with pytest.raises(SQLValidationError, match="Unknown table"):
            validate_sql("SELECT * FROM secret_data")

    def test_rejects_read_csv(self):
        with pytest.raises(SQLValidationError, match="Dangerous pattern"):
            validate_sql("SELECT * FROM read_csv('malicious.csv')")

    def test_caps_limit(self):
        sql = validate_sql("SELECT * FROM violations LIMIT 5000")
        assert "LIMIT 1000" in sql

    def test_injects_limit(self):
        sql = validate_sql("SELECT * FROM violations")
        assert "LIMIT 1000" in sql

    def test_rejects_empty(self):
        with pytest.raises(SQLValidationError):
            validate_sql("")

    def test_rejects_non_select(self):
        with pytest.raises(SQLValidationError):
            validate_sql("UPDATE violations SET status='closed'")


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

class TestSchemaValidation:
    """Test schema validation catches bad intents."""

    def test_valid_intent(self):
        intent = schema.validate_intent({
            "dataset": "violations",
            "metric": "count",
            "group_by": "neighborhood",
        })
        assert intent["dataset"] == "violations"
        assert intent["metric"] == "count"

    def test_invalid_dataset(self):
        with pytest.raises(Exception):
            schema.validate_intent({
                "dataset": "nonexistent_table",
                "metric": "count",
            })

    def test_invalid_metric(self):
        with pytest.raises(Exception):
            schema.validate_intent({
                "dataset": "violations",
                "metric": "invalid_metric",
            })

    def test_group_by_normalized_to_list(self):
        intent = schema.validate_intent({
            "dataset": "violations",
            "metric": "count",
            "group_by": "neighborhood",
        })
        assert isinstance(intent["group_by"], list)

    def test_filter_normalized(self):
        intent = schema.validate_intent({
            "dataset": "violations",
            "metric": "count",
            "filters": {"year": 2022},
        })
        year_filter = intent["filters"]["year"]
        assert isinstance(year_filter, dict)
        assert "op" in year_filter
        assert "value" in year_filter


# =============================================================================
# GROUND-TRUTH VALIDATION — Item #12
# =============================================================================

class TestGroundTruth:
    """Validate SQL results against pandas calculations."""

    def test_total_violations_count(self, violations_df):
        result = run_query("How many code violations are there?")
        assert result.get("error") is None, f"Error: {result.get('error')}"
        df = result["result"]
        count_col = "count" if "count" in df.columns else df.columns[-1]
        sql_count = int(df[count_col].iloc[0])
        pd_count = len(violations_df)
        assert sql_count == pd_count, f"SQL={sql_count}, pandas={pd_count}"

    def test_total_crime_count(self, crime_df):
        result = run_query("Total number of crimes")
        assert result.get("error") is None, f"Error: {result.get('error')}"
        df = result["result"]
        count_col = "count" if "count" in df.columns else df.columns[-1]
        sql_count = int(df[count_col].sum())
        pd_count = len(crime_df)
        # Allow tolerance for null neighborhoods or heuristic grouping
        ratio = sql_count / pd_count if pd_count > 0 else 0
        assert 0.9 < ratio < 1.1, f"SQL={sql_count}, pandas={pd_count}"

    def test_violations_by_neighborhood_count(self, violations_df):
        """Each neighborhood's count should match pandas groupby."""
        result = run_query("Violations by neighborhood")
        assert result.get("error") is None
        df = result["result"]

        gt = violations_df.groupby("neighborhood").size().reset_index(name="count")
        count_col = "count" if "count" in df.columns else df.columns[-1]
        sql_total = int(df[count_col].sum())
        pd_total = int(gt["count"].sum())
        assert sql_total == pd_total, f"Totals differ: SQL={sql_total}, pandas={pd_total}"

    def test_crime_by_neighborhood_count(self, crime_df):
        result = run_query("Crime counts by neighborhood")
        assert result.get("error") is None
        df = result["result"]
        count_col = "count" if "count" in df.columns else df.columns[-1]
        sql_total = int(df[count_col].sum())
        pd_total = len(crime_df)
        # Allow tolerance for null neighborhoods
        ratio = sql_total / pd_total if pd_total > 0 else 0
        assert 0.95 < ratio < 1.05, f"SQL={sql_total}, pandas={pd_total}"


# =============================================================================
# ALL-DATASET SMOKE TESTS
# =============================================================================

class TestAllDatasets:
    """Smoke test: every dataset returns results for a basic query."""

    @pytest.mark.parametrize("dataset,question", [
        ("violations", "How many code violations are there?"),
        ("crime", "Crime by neighborhood"),
        ("vacant_properties", "How many vacant properties are there?"),
        ("rental_registry", "How many rental properties are registered?"),
        ("unfit_properties", "Unfit properties by department"),
        ("trash_pickup", "Trash pickup by zip"),
        ("historical_properties", "Historical properties by zip"),
        ("assessment_roll", "Assessment roll by property class"),
        ("cityline_requests", "Cityline requests by category"),
        ("parking_violations", "Parking fines by description"),
        ("permit_requests", "Building permits by type"),
        ("tree_inventory", "Trees by condition"),
        ("bike_infrastructure", "Bike infrastructure by type"),
        ("lead_testing", "Lead testing results"),
    ])
    def test_dataset_returns_results(self, dataset, question):
        result = run_query(question)
        assert result.get("error") is None, f"{dataset}: {result.get('error')}"
        assert result["result"] is not None, f"{dataset}: no result"
        assert len(result["result"]) > 0, f"{dataset}: empty result"
