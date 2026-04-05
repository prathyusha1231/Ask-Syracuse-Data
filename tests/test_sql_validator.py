"""
Tests for SQL validator guardrails.
Verifies that dangerous SQL is rejected and safe SQL is accepted.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.sql_validator import validate_sql, SQLValidationError


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
    print("SQL VALIDATOR TESTS")
    print("=" * 60)

    # --- Tests that SHOULD pass (valid SQL) ---

    def test_simple_select():
        result = validate_sql("SELECT * FROM violations")
        assert "SELECT" in result and "LIMIT" in result

    def test_with_cte():
        sql = "WITH cte AS (SELECT * FROM crime) SELECT * FROM cte"
        result = validate_sql(sql)
        assert "LIMIT" in result

    def test_existing_limit_preserved():
        result = validate_sql("SELECT * FROM violations LIMIT 50")
        assert "LIMIT 50" in result

    def test_high_limit_capped():
        result = validate_sql("SELECT * FROM violations LIMIT 5000")
        assert "LIMIT 1000" in result

    def test_multiple_tables_join():
        sql = "SELECT v.*, r.* FROM violations v JOIN rental_registry r ON v.sbl = r.sbl"
        result = validate_sql(sql)
        assert "LIMIT" in result

    def test_markdown_fences_stripped():
        sql = "```sql\nSELECT * FROM crime\n```"
        result = validate_sql(sql)
        assert "SELECT" in result and "```" not in result

    check("simple SELECT", test_simple_select)
    check("WITH CTE", test_with_cte)
    check("existing LIMIT preserved", test_existing_limit_preserved)
    check("high LIMIT capped to 1000", test_high_limit_capped)
    check("multi-table JOIN", test_multiple_tables_join)
    check("markdown fences stripped", test_markdown_fences_stripped)

    # --- Tests that SHOULD fail (dangerous SQL) ---

    def test_reject_insert():
        try:
            validate_sql("INSERT INTO violations VALUES (1, 2, 3)")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_drop():
        try:
            validate_sql("DROP TABLE violations")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_delete():
        try:
            validate_sql("DELETE FROM violations WHERE 1=1")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_update():
        try:
            validate_sql("UPDATE violations SET status='closed'")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_read_csv():
        try:
            validate_sql("SELECT * FROM read_csv('malicious.csv')")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_read_parquet():
        try:
            validate_sql("SELECT * FROM read_parquet('data.parquet')")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_httpfs():
        try:
            validate_sql("SELECT * FROM httpfs('http://evil.com/data.csv')")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_unknown_table():
        try:
            validate_sql("SELECT * FROM secret_table")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_multi_statement():
        try:
            validate_sql("SELECT * FROM violations; DROP TABLE crime")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_empty():
        try:
            validate_sql("")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_pg_catalog():
        try:
            validate_sql("SELECT * FROM pg_catalog.pg_tables")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_load_extension():
        try:
            validate_sql("SELECT load_extension('httpfs')")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_create_table():
        try:
            validate_sql("CREATE TABLE evil AS SELECT * FROM violations")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    def test_reject_alter_table():
        try:
            validate_sql("ALTER TABLE violations ADD COLUMN evil TEXT")
            assert False, "Should have raised SQLValidationError"
        except SQLValidationError:
            pass

    check("reject INSERT", test_reject_insert)
    check("reject DROP", test_reject_drop)
    check("reject DELETE", test_reject_delete)
    check("reject UPDATE", test_reject_update)
    check("reject read_csv", test_reject_read_csv)
    check("reject read_parquet", test_reject_read_parquet)
    check("reject httpfs", test_reject_httpfs)
    check("reject unknown table", test_reject_unknown_table)
    check("reject multi-statement", test_reject_multi_statement)
    check("reject empty SQL", test_reject_empty)
    check("reject pg_catalog", test_reject_pg_catalog)
    check("reject load_extension", test_reject_load_extension)
    check("reject CREATE TABLE", test_reject_create_table)
    check("reject ALTER TABLE", test_reject_alter_table)

    print()
    print("=" * 60)
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
