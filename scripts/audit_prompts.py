#!/usr/bin/env python3
"""Audit every distinct level 2-3 prompt built from real data before the pilot.

    python scripts/audit_prompts.py                      # rubric decisions, levels 2 and 3
    python scripts/audit_prompts.py --decisions EN-3b --levels 3

Prompts are built exactly as `run_experiment.py estimate` builds them (same sources,
context selection and disguise). For each distinct prompt (decision, level, scenario,
stake; repeats only rotate the options) it prints the independent audit's findings
(llmtrust.disguise.audit_text). When the prompt builder refuses a prompt with
DisguiseLeak, the findings are listed per source document so the leak can be found.
No model is called. Exit 1 if any prompt has findings or cannot be built, else 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llmtrust.cases import load_case  # noqa: E402
from llmtrust.disguise import audit_text, disguise_text, load_rules  # noqa: E402
from llmtrust.prompts import DisguiseLeak  # noqa: E402
from llmtrust.runner import trial_grid  # noqa: E402
from llmtrust.score import load_rubric  # noqa: E402
from llmtrust.sources import load_sources, select_context  # noqa: E402

import run_experiment  # noqa: E402

AUDIT_ARM = "audit"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", type=Path, default=ROOT / "cases" / "enron.yaml")
    p.add_argument("--rules", type=Path, default=ROOT / "cases" / "enron.disguise.yaml")
    p.add_argument("--rubric", type=Path, default=ROOT / "cases" / "enron.rubric.yaml")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--decisions", nargs="+", default=None,
                   help="default: the decisions that have an entry in the rubric")
    p.add_argument("--levels", nargs="+", type=int, default=[2, 3], choices=[2, 3])
    return p


def _locate(case, dp, data_root, spec, rules) -> list[str]:
    """Per-document audit findings, to show where a refused prompt leaks."""
    docs = load_sources(case, dp, data_root, level=spec.level, scenario=spec.scenario)
    docs = select_context(docs, run_experiment.MAX_CONTEXT_CHARS)[0]
    lines = []
    parts = [("role", dp.role), ("scenario brief", dp.scenario_brief),
             ("stake statement", dp.stake_statement)]
    parts += [(f"option {o!r}", o) for o in dp.options]
    parts += [(f"{d.kind} {d.date} {d.title!r}", d.title + "\n" + d.text) for d in docs]
    for label, text in parts:
        out = disguise_text(text, spec.level, rules)
        hits = audit_text(out, rules, spec.level)
        if hits:
            lines.append(f"    {label}: {hits}")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = load_case(args.case)
    rules = load_rules(args.rules)
    decisions = (frozenset(args.decisions) if args.decisions
                 else frozenset(load_rubric(args.rubric).options))
    specs = trial_grid(case, [AUDIT_ARM], 1, decisions=decisions)
    dps = {dp.id: dp for dp in case.decision_points}

    setup = run_experiment.Setup(case=case, dps=dps, arms={}, caps={}, rules=rules,
                                 out_dir=Path("."), data_root=Path(args.data_root),
                                 budget=None)  # type: ignore[arg-type]
    builder = run_experiment.PromptBuilder(setup)

    bad = 0
    total = 0
    for spec in specs:
        if spec.level not in args.levels:
            continue
        total += 1
        label = (f"{spec.decision_id} L{spec.level} {spec.scenario} "
                 f"{'stake' if spec.stake else 'nostake'}")
        try:
            prompt = builder.build(spec)
        except DisguiseLeak as exc:
            bad += 1
            print(f"{label}: REFUSED: {exc}")
            for line in _locate(case, dps[spec.decision_id], Path(args.data_root), spec, rules):
                print(line)
            continue
        except Exception as exc:
            bad += 1
            print(f"{label}: cannot build: {type(exc).__name__}: {exc}")
            continue
        hits = audit_text(prompt.system + "\n" + prompt.user, rules, spec.level)
        if hits:
            bad += 1
        print(f"{label}: {hits if hits else 'clean'}")
    print(f"{total} prompts audited, {bad} with findings or errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
