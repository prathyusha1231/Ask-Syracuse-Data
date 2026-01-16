"""
Intent parsing layer: converts natural language questions into structured intents.
LLM is used only for parsing; no data access or analysis occurs here.
"""
from __future__ import annotations
import json
from typing import Any, Callable, Dict, Optional

from llm.prompt_templates import NL_TO_INTENT_PROMPT, NL_TO_JOIN_INTENT_PROMPT

LLMCallable = Callable[[str], str]


class IntentParsingError(Exception):
    """Raised when an intent cannot be parsed or validated."""


def _is_join_query(question: str) -> bool:
    """Detect if a question requires a cross-dataset join."""
    q = question.lower()

    # Patterns indicating multiple datasets
    join_patterns = [
        # rental + violations
        ("rental" in q and "violation" in q),
        ("rental" in q and "code" in q and "violation" in q),
        ("propert" in q and "violation" in q),
        # vacant + violations
        ("vacant" in q and "violation" in q),
        # rental + vacant
        ("rental" in q and "vacant" in q),
        # Generic join indicators
        ("with" in q and any(d in q for d in ["violation", "vacant", "rental"])),
        ("properties" in q and "violation" in q),
    ]

    return any(join_patterns)


def _heuristic_join_intent(question: str) -> Optional[Dict[str, Any]]:
    """Deterministic fallback for common join queries when LLM is unavailable."""
    q = question.lower()

    # rental + violations (most common expected query)
    if ("rental" in q or "propert" in q) and "violation" in q:
        # Determine join type based on keywords
        join_type = "sbl" if ("specific" in q or "property" in q or "address" in q) else "zip"
        group_by = "zip" if join_type == "zip" else None
        return {
            "query_type": "join",
            "primary_dataset": "rental_registry",
            "secondary_dataset": "violations",
            "join_type": join_type,
            "metric": "count",
            "group_by": group_by,
            "filters": {},
            "limit": 20 if join_type == "sbl" else None,
        }

    # vacant + violations
    if "vacant" in q and "violation" in q:
        join_type = "sbl" if ("specific" in q or "property" in q) else "zip"
        group_by = "zip" if join_type == "zip" else "neighborhood"
        return {
            "query_type": "join",
            "primary_dataset": "vacant_properties",
            "secondary_dataset": "violations",
            "join_type": join_type,
            "metric": "count",
            "group_by": group_by,
            "filters": {},
            "limit": None,
        }

    # rental + vacant
    if "rental" in q and "vacant" in q:
        return {
            "query_type": "join",
            "primary_dataset": "rental_registry",
            "secondary_dataset": "vacant_properties",
            "join_type": "sbl",
            "metric": "count",
            "group_by": None,
            "filters": {},
            "limit": 20,
        }

    return None


def _heuristic_intent(question: str) -> Optional[Dict[str, Any]]:
    """Simple deterministic fallback for common single-dataset questions."""
    q = question.lower()

    # Check for join queries first
    if _is_join_query(q):
        return _heuristic_join_intent(question)

    # Single-dataset queries
    if "violation" in q:
        group_by = "neighborhood" if "neighborhood" in q else None
        if "zip" in q:
            group_by = "complaint_zip"
        return {"dataset": "violations", "metric": "count", "group_by": group_by, "filters": {}}

    if "vacant" in q:
        group_by = "neighborhood" if "neighborhood" in q else None
        if "zip" in q:
            group_by = "zip"
        return {"dataset": "vacant_properties", "metric": "count", "group_by": group_by, "filters": {}}

    if "rental" in q:
        group_by = "zip" if "zip" in q else None
        return {"dataset": "rental_registry", "metric": "count", "group_by": group_by, "filters": {}}

    if "crime" in q or "offense" in q:
        return {"dataset": "crime_2022", "metric": "count", "group_by": "code_defined", "filters": {}}

    return None


def parse_intent(question: str, llm: Optional[LLMCallable] = None) -> Dict[str, Any]:
    """
    Convert a natural language question into a structured intent dict.
    If an LLM callable is provided, it must return a JSON string.
    A deterministic heuristic is used as a fallback when no LLM is supplied.
    """
    if not question or not question.strip():
        raise IntentParsingError("Question is empty or missing.")

    intent: Optional[Dict[str, Any]] = None

    if llm:
        # Detect if this is a join query
        is_join = _is_join_query(question)
        prompt_template = NL_TO_JOIN_INTENT_PROMPT if is_join else NL_TO_INTENT_PROMPT
        prompt = prompt_template + f"\nQuestion: {question}\nJSON:"
        raw = llm(prompt)
        try:
            intent = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IntentParsingError(f"LLM did not return valid JSON: {exc}") from exc
    else:
        intent = _heuristic_intent(question)
        if intent is None:
            raise IntentParsingError("Unable to parse intent without LLM; question unsupported.")

    return intent


__all__ = ["parse_intent", "IntentParsingError", "_is_join_query"]
