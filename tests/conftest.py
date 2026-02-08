"""
Pytest configuration for Ask Syracuse Data tests.
Forces heuristic-only mode (no LLM) unless FORCE_LLM_TESTS=1 is set.
"""
import os
import pytest
from unittest.mock import patch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "llm: marks tests that require LLM API access")


@pytest.fixture(autouse=True, scope="session")
def disable_llm_for_tests():
    """Patch load_api_key to return None, forcing heuristic mode.
    Set FORCE_LLM_TESTS=1 to use the real API key."""
    if os.environ.get("FORCE_LLM_TESTS") == "1":
        yield
    else:
        with patch("pipeline.main.load_api_key", return_value=None):
            yield
