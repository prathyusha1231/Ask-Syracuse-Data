"""
Intent parsing layer: converts natural language questions into structured intents.
LLM is used only for parsing; no data access or analysis occurs here.
"""
from __future__ import annotations
import json
from typing import Any, Callable, Dict, Optional

from llm.prompt_templates import NL_TO_INTENT_PROMPT

LLMCallable = Callable[[str], str]


class IntentParsingError(Exception):
    """Raised when an intent cannot be parsed or validated."""


def _heuristic_intent(question: str) -> Optional[Dict[str, Any]]:
    """Simple deterministic fallback for common questions when LLM is unavailable."""
    q = question.lower()
    if "violation" in q and "neighborhood" in q:
        return {"dataset": "violations", "metric": "count", "group_by": "neighborhood", "filters": {}}
    if "vacant" in q and "neighborhood" in q:
        return {"dataset": "vacant_properties", "metric": "count", "group_by": "neighborhood", "filters": {}}
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
        prompt = NL_TO_INTENT_PROMPT + f"\nQuestion: {question}\nJSON:"
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


__all__ = ["parse_intent", "IntentParsingError"]
