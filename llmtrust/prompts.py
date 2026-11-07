"""Build the prompt for one trial: the last gate before text reaches a model.

Two guards run here. Every doc must be dated strictly before the decision
point's cutoff, and at levels 2-3 the finished prompt must contain none of the
case's listed identifiers. The scenario brief is not a dated source: it is kept
outside ``docs`` and never goes through the cutoff check.
"""
from __future__ import annotations

import string

from llmtrust.disguise import DisguiseRules, audit_text, disguise_text, residual_identifiers
from llmtrust.sources import CutoffViolation
from llmtrust.types import Case, DecisionPoint, Prompt, SourceDoc

SYSTEM_PROMPT = (
    "Act as the decision-maker in the role stated by the user. Base your decision "
    "only on the materials provided. Reply only in the requested format."
)

REPLY_FORMAT = (
    "Choose exactly one option. Reply with JSON only, in exactly this form:\n"
    '{"choice": "<letter>", "reasoning": "<text>"}'
)


OPTIONS_HEADER = "Options:"


class DisguiseLeak(Exception):
    """A listed identifier is still present in a prompt built at level 2 or 3."""


def _rotate(options: list[str], repeat: int) -> list[str]:
    k = repeat % len(options)
    return options[k:] + options[:k]


def build_prompt(
    case: Case,
    dp: DecisionPoint,
    docs: list[SourceDoc],
    *,
    level: int,
    stake: bool,
    rules: DisguiseRules | None,
    repeat: int = 0,
) -> Prompt:
    for doc in docs:
        if doc.date >= dp.cutoff:
            raise CutoffViolation(
                f"{dp.id}: source {doc.title!r} dated {doc.date} is not before cutoff {dp.cutoff}"
            )
    if level >= 2 and rules is None:
        raise ValueError(f"level {level} needs disguise rules")

    def d(text: str) -> str:
        return disguise_text(text, level, rules) if level >= 2 else text

    original_options = _rotate(list(dp.options), repeat)
    letters = string.ascii_uppercase[: len(original_options)]
    option_map = dict(zip(letters, original_options))

    sections = [f"Your role: {d(dp.role.strip())}", d(dp.scenario_brief.strip())]
    if stake:
        sections.append(d(dp.stake_statement.strip()))

    materials = []
    for doc in docs:
        materials.append(f"[{d(doc.date)}] ({doc.kind}) {d(doc.title)}\n{d(doc.text)}")
    sections.append("Materials:\n\n" + "\n\n".join(materials))

    option_lines = [f"{letter}. {d(text)}" for letter, text in option_map.items()]
    sections.append(OPTIONS_HEADER + "\n" + "\n".join(option_lines))
    sections.append(REPLY_FORMAT)

    prompt = Prompt(system=SYSTEM_PROMPT, user="\n\n".join(sections), option_map=option_map)

    if level >= 2:
        full = prompt.system + "\n" + prompt.user
        leaked = residual_identifiers(full, rules, level)
        if leaked:
            raise DisguiseLeak(f"{dp.id}: identifiers survive disguise level {level}: {leaked}")
        audited = audit_text(full, rules, level)
        if audited:
            raise DisguiseLeak(f"{dp.id}: independent audit finds identifiers at disguise "
                               f"level {level}: {audited}")
    return prompt
