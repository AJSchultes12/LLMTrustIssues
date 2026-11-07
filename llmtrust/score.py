from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OptionScore:
    matches_executives: bool
    alignment: int          # 0, 1 or 2; see docs/rubric.md
    escalation: bool


@dataclass(frozen=True)
class Rubric:
    options: dict[str, dict[str, OptionScore]]   # decision id -> option text -> score
    reasoning_criteria: list[str]


def load_rubric(path: Path) -> Rubric:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    options: dict[str, dict[str, OptionScore]] = {}
    for did, opts in raw["decisions"].items():
        options[did] = {}
        for text, fields in opts.items():
            alignment = fields["alignment"]
            if isinstance(alignment, bool) or alignment not in (0, 1, 2):
                raise ValueError(
                    f"{did}: alignment must be 0, 1 or 2, got {alignment!r} for option {text!r}"
                )
            options[did][text] = OptionScore(
                matches_executives=bool(fields["matches_executives"]),
                alignment=alignment,
                escalation=bool(fields["escalation"]),
            )
    return Rubric(options=options, reasoning_criteria=list(raw["reasoning_criteria"]))
