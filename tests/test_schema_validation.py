"""
Tests for schema validation edge cases — join intents, filter validation, BETWEEN bounds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.schema import validate_intent, validate_join_intent, get_join_config


def run_tests():
    passed = 0
    failed = 0
    total = 0

    def check(name, fn):
        nonlocal passed, failed, total
        total += 1
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {name} — {e}")

    print("=" * 60)
    print("SCHEMA VALIDATION TESTS")
    print("=" * 60)

    # --- Single-dataset intent validation ---

    def test_valid_count_intent():
        intent = {"dataset": "violations", "metric": "count"}
        result = validate_intent(intent)
        assert result["metric"] == "count"

    def test_invalid_dataset():
        try:
            validate_intent({"dataset": "nonexistent", "metric": "count"})
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e).lower() or "unknown" in str(e).lower()

    def test_invalid_metric():
        try:
            validate_intent({"dataset": "violations", "metric": "median"})
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_valid_filter():
        intent = {
            "dataset": "violations",
            "metric": "count",
            "filters": {"year": {"op": ">=", "value": 2020}},
        }
        result = validate_intent(intent)
        assert result["filters"]["year"]["op"] == ">="

    def test_between_filter_valid():
        intent = {
            "dataset": "violations",
            "metric": "count",
            "filters": {"year": {"op": "between", "value": [2020, 2023]}},
        }
        result = validate_intent(intent)
        assert result["filters"]["year"]["value"] == [2020, 2023]

    def test_between_filter_reversed_bounds():
        try:
            validate_intent({
                "dataset": "violations",
                "metric": "count",
                "filters": {"year": {"op": "between", "value": [2023, 2020]}},
            })
            assert False, "Should raise ValueError for reversed bounds"
        except ValueError as e:
            assert "reversed" in str(e).lower()

    def test_invalid_filter_column():
        try:
            validate_intent({
                "dataset": "violations",
                "metric": "count",
                "filters": {"nonexistent_col": {"op": "=", "value": "test"}},
            })
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_group_by_normalized_to_list():
        intent = {
            "dataset": "violations",
            "metric": "count",
            "group_by": "neighborhood",
        }
        result = validate_intent(intent)
        assert isinstance(result["group_by"], list)

    check("valid count intent", test_valid_count_intent)
    check("invalid dataset rejected", test_invalid_dataset)
    check("invalid metric rejected", test_invalid_metric)
    check("valid filter accepted", test_valid_filter)
    check("BETWEEN filter valid bounds", test_between_filter_valid)
    check("BETWEEN filter reversed bounds rejected", test_between_filter_reversed_bounds)
    check("invalid filter column rejected", test_invalid_filter_column)
    check("group_by normalized to list", test_group_by_normalized_to_list)

    # --- Join intent validation ---

    def test_valid_join_config():
        config = get_join_config("violations", "rental_registry")
        assert config is not None

    def test_invalid_join_pair():
        try:
            get_join_config("tree_inventory", "lead_testing")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_valid_join_intent():
        intent = {
            "query_type": "join",
            "primary_dataset": "violations",
            "secondary_dataset": "rental_registry",
            "metric": "count",
        }
        result = validate_join_intent(intent)
        assert result["primary_dataset"] == "violations"

    def test_join_missing_secondary():
        try:
            validate_join_intent({
                "query_type": "join",
                "primary_dataset": "violations",
                "metric": "count",
            })
            assert False, "Should raise ValueError/KeyError"
        except (ValueError, KeyError):
            pass

    check("valid join config lookup", test_valid_join_config)
    check("invalid join pair rejected", test_invalid_join_pair)
    check("valid join intent", test_valid_join_intent)
    check("join missing secondary rejected", test_join_missing_secondary)

    print()
    print("=" * 60)
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
