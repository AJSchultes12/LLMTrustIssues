"""Rewrite a document in different words, keeping every fact unchanged.

Used on the Watkins memo so a model cannot recognise it by its wording. Names
and numbers must survive untouched: later disguise steps replace them.
"""
from __future__ import annotations

from llmtrust.budget import estimate_tokens
from llmtrust.models import ModelClient

REWRITE_SYSTEM = (
    "You rewrite documents in different words without changing what they say.\n"
    "Rules:\n"
    "- Keep every factual claim. Do not add, remove, soften or strengthen any claim.\n"
    "- Keep every number, date, name of a person, name of a company and name of a "
    "financial instrument exactly as written. Copy them character for character.\n"
    "- Change only the wording and the sentence structure.\n"
    "- Keep the length within about 15% of the original.\n"
    "- Output only the rewritten text, with no preface, title, note or commentary."
)


def default_max_tokens(text: str) -> int:
    return max(2048, 2 * estimate_tokens(text))


def rewrite_document(text: str, client: ModelClient, *, max_tokens: int | None = None) -> str:
    if max_tokens is None:
        max_tokens = default_max_tokens(text)
    completion = client.complete(REWRITE_SYSTEM, text, max_tokens=max_tokens)
    return completion.text.strip()
