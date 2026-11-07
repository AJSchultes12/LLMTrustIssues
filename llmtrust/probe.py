"""Recognition probe: can the model name the disguised company?

The probe shows a model the same materials as the decision prompt, minus the
decision question, and asks which company this is, which year, and how it ended.
If the reply names a real identifier the trial counts as contaminated. Probing is
for levels 2 and 3 only; at level 1 the company is named anyway. For the twin
scenario the same check shows whether the model names the real case.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from llmtrust.budget import Budget
from llmtrust.disguise import DisguiseRules, residual_identifiers
from llmtrust.models import Arm, ModelClient
from llmtrust.prompts import OPTIONS_HEADER
from llmtrust.runner import _write_atomic, charged_complete
from llmtrust.types import Prompt, TrialSpec

PROBE_SYSTEM = "You are a careful business historian."

PROBE_QUESTION = (
    "Based only on the situation described above, which company or organisation does "
    "this describe, in which year it took place, and how the situation ended? "
    "If you cannot tell, say so.\n"
    "Reply with JSON only, in exactly this form:\n"
    '{"company": "...", "year": "...", "outcome": "..."}'
)


def probe_key(spec: TrialSpec) -> str:
    """One probe per case, decision, level, scenario and arm; stake and repeat share it."""
    return f"{spec.case_id}.{spec.decision_id}.L{spec.level}.{spec.scenario}.{spec.arm}"


def build_probe_prompt(prompt: Prompt) -> Prompt:
    """Keep the user text up to the options section, drop the options and reply format."""
    marker = "\n\n" + OPTIONS_HEADER + "\n"
    idx = prompt.user.rfind(marker)
    if idx < 0:
        raise ValueError("prompt has no options section to cut at")
    user = prompt.user[:idx] + "\n\n" + PROBE_QUESTION
    return Prompt(system=PROBE_SYSTEM, user=user, option_map={})


def is_identified(answer: str, rules: DisguiseRules) -> bool:
    """True when the reply contains any real identifier listed in the disguise rules.

    Uses the disguise's own matching (word boundary; case-insensitive for keys longer
    than 4 characters, case-sensitive for shorter ones). This can over-flag, for example
    a place name the model mentions innocently. That is the conservative direction: a
    trial wrongly marked contaminated is excluded, a contaminated one is never kept.
    """
    return bool(residual_identifiers(answer, rules, 2))


def run_probe(spec: TrialSpec, prompt: Prompt, *, client: ModelClient, arm: Arm,
              budget: Budget, rules: DisguiseRules, out_dir: Path,
              max_tokens: int = 300) -> Path:
    if spec.level == 1:
        raise ValueError("probing is for levels 2 and 3 only")
    key = probe_key(spec)
    path = Path(out_dir) / "probes" / f"{key}.json"
    if path.exists():
        return path

    probe = build_probe_prompt(prompt)
    completion = charged_complete(client, arm, budget, probe.system, probe.user, max_tokens)

    record = {
        "probe_key": key,
        "case_id": spec.case_id,
        "decision_id": spec.decision_id,
        "level": spec.level,
        "scenario": spec.scenario,
        "arm": spec.arm,
        "prompt": {"system": probe.system, "user": probe.user},
        "completion": asdict(completion),
        "identified": is_identified(completion.text, rules),
        "finished": completion.finished_at[:10],
    }
    _write_atomic(path, record)
    return path
