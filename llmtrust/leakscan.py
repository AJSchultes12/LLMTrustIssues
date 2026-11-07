"""Outcome-term scan: flag outcome words in a model's reasoning that its prompt never held.

A term that the prompt itself contained (for example "Raptor" in an allowed memo) is
not a leak: the model could have read it there. Only terms the model brought in
unprompted count.
"""
from __future__ import annotations

import re

_LEFT = r"(?<![A-Za-z0-9])"
_RIGHT = r"(?![A-Za-z0-9])"


def _contains(text: str, term: str) -> bool:
    return re.search(f"{_LEFT}{re.escape(term)}{_RIGHT}", text, re.IGNORECASE) is not None


def scan_reasoning(reasoning: str, prompt_text: str, terms: list[str]) -> list[str]:
    """Terms (in ``terms`` order, each once) found in ``reasoning`` but not in ``prompt_text``.

    Matching is case-insensitive and whole-word, so "bankruptcy" does not match inside
    "nonbankruptcyish" but does match "bankruptcy's".
    """
    flagged: list[str] = []
    for term in terms:
        if not term or term in flagged:
            continue
        if _contains(reasoning, term) and not _contains(prompt_text, term):
            flagged.append(term)
    return flagged
