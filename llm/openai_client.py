"""
OpenAI client wrapper for intent parsing (JSON-only responses).
Reads API key from environment or .env file at repo root.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_api_key() -> str | None:
    """Load OPENAI_API_KEY from environment or .env."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"' ")
    return None


def make_openai_intent_llm(model: str = "gpt-4o-mini") -> Callable[[str], str]:
    """
    Return a callable(prompt) -> JSON string using OpenAI Chat Completions.
    The caller must ensure prompts ask for JSON; response_format enforces JSON object.
    """
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

    client = OpenAI(api_key=api_key)

    def _llm(prompt: str) -> str:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an intent parser. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        return completion.choices[0].message.content

    return _llm


def make_openai_sql_llm(model: str = "gpt-4o-mini") -> Callable[[str], str]:
    """
    Return a callable(prompt) -> SQL string using OpenAI Chat Completions.
    Unlike the intent LLM, this does NOT enforce JSON response format.
    """
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

    client = OpenAI(api_key=api_key)

    def _llm(prompt: str) -> str:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a SQL query generator. Return only valid DuckDB SQL. No explanations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        return completion.choices[0].message.content

    return _llm


__all__ = ["make_openai_intent_llm", "make_openai_sql_llm", "load_api_key"]
