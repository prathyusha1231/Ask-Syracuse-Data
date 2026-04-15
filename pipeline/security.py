"""
Prompt-injection detection and response helpers for Ask Syracuse Data.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


BLOCKED_USER_MESSAGE = (
    "Your request appears to contain instructions aimed at manipulating the "
    "assistant or accessing internal system details rather than querying "
    "Syracuse data. Please ask a data question such as "
    "\"violations by neighborhood\" or \"crime by year\"."
)

SUSPICIOUS_LIMITATION_NOTE = (
    "Instruction-like text in the request was ignored for safety, and the "
    "query was processed using the guarded heuristic path."
)


@dataclass(frozen=True)
class PromptInjectionAssessment:
    """Result of classifying a user request for prompt-injection risk."""

    status: str
    category: str | None = None
    reason: str | None = None
    user_message: str | None = None

    def to_metadata(self) -> dict:
        """Serialize the assessment into response metadata."""
        return {
            "status": self.status,
            "category": self.category,
            "reason": self.reason,
        }


_BLOCK_PATTERNS = [
    (
        "prompt_exfiltration",
        r"\b(reveal|show|print|dump|display|expose)\b.{0,50}\b(system prompt|developer prompt|hidden prompt|internal prompt)\b",
        "attempt to extract protected prompts",
    ),
    (
        "secret_exfiltration",
        r"\b(reveal|show|print|dump|display|read)\b.{0,50}\b(api key|openai_api_key|token|secret|password|credential|\.env)\b",
        "attempt to extract secrets or configuration",
    ),
    (
        "unsafe_execution",
        r"\b(drop table|delete from|truncate table|insert into|update\s+\w+\s+set|read_csv|read_parquet|load_extension|attach\b|detach\b|httpfs|http_get|information_schema|pg_catalog)\b",
        "attempt to trigger unsafe SQL or external access",
    ),
]

_SUSPICIOUS_PATTERNS = [
    (
        "instruction_override",
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,50}\b(previous instructions|instructions|rules|guardrails|safety|policy)\b",
        "attempt to override system instructions",
    ),
    (
        "role_override",
        r"\b(act as|pretend to be|roleplay as)\b.{0,40}\b(system|developer|administrator|admin)\b",
        "attempt to override the assistant role",
    ),
    (
        "jailbreak_language",
        r"\b(jailbreak|do anything now|dan)\b",
        "attempt to jailbreak the assistant",
    ),
]


def assess_prompt_injection(question: str) -> PromptInjectionAssessment:
    """Classify prompt-injection risk for a user question."""
    if not question or not question.strip():
        return PromptInjectionAssessment(status="safe")

    normalized = re.sub(r"\s+", " ", question.strip().lower())

    for category, pattern, reason in _BLOCK_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return PromptInjectionAssessment(
                status="blocked",
                category=category,
                reason=reason,
                user_message=BLOCKED_USER_MESSAGE,
            )

    for category, pattern, reason in _SUSPICIOUS_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return PromptInjectionAssessment(
                status="suspicious",
                category=category,
                reason=reason,
            )

    return PromptInjectionAssessment(status="safe")


__all__ = [
    "BLOCKED_USER_MESSAGE",
    "SUSPICIOUS_LIMITATION_NOTE",
    "PromptInjectionAssessment",
    "assess_prompt_injection",
]
