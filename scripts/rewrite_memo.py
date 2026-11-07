#!/usr/bin/env python3
"""Rewrite the manual documents of a decision point in different words.

Reads data/raw/<case_id>/<decision>/documents/*.txt and writes each rewrite to
documents_rewritten/<same name>. Existing rewrites are skipped unless --force.
Spend is checked against the arm's pool cap before every call.

This script NEVER creates the REVIEWED marker. After running it, read each
rewrite against the original, edit if needed, then create REVIEWED yourself.

Usage:
    python scripts/rewrite_memo.py --case cases/enron.yaml --decision EN-3b --arm claude
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmtrust.budget import Budget, BudgetExceeded, estimate_tokens  # noqa: E402
from llmtrust.cases import load_case  # noqa: E402
from llmtrust.models import Arm, ModelClient, load_arms, make_client  # noqa: E402
from llmtrust.rewrite import rewrite_document  # noqa: E402
from llmtrust.types import Completion  # noqa: E402


class _MeteredClient:
    """Checks the budget before each call and records actual spend after it."""

    def __init__(self, inner: ModelClient, arm: Arm, budget: Budget):
        self.inner, self.arm, self.budget = inner, arm, budget

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        estimate = self.budget.estimate_usd(
            self.arm, estimate_tokens(system) + estimate_tokens(user), max_tokens)
        self.budget.check(self.arm.pool, estimate)
        completion = self.inner.complete(system, user, max_tokens=max_tokens)
        self.budget.record(self.arm.pool, self.budget.estimate_usd(
            self.arm, completion.input_tokens, completion.output_tokens))
        return completion


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", required=True, type=Path)
    p.add_argument("--decision", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    p.add_argument("--budget-state", type=Path, default=Path("data/runs/budget.json"))
    p.add_argument("--force", action="store_true", help="overwrite existing rewrites")
    return p


def main(argv: list[str] | None = None,
         client_factory: Callable[[Arm], ModelClient] = make_client) -> int:
    args = _build_parser().parse_args(argv)

    case = load_case(args.case)
    if args.decision not in {dp.id for dp in case.decision_points}:
        print(f"error: decision {args.decision!r} not in {args.case}", file=sys.stderr)
        return 2
    arms = load_arms(args.models)
    if args.arm not in arms:
        print(f"error: arm {args.arm!r} not in {args.models}; have {sorted(arms)}",
              file=sys.stderr)
        return 2
    arm = arms[args.arm]
    caps = (yaml.safe_load(args.models.read_text(encoding="utf-8")) or {}).get("caps") or {}

    base = args.data_root / "raw" / case.case_id / args.decision
    sources = sorted((base / "documents").glob("*.txt"))
    if not sources:
        print(f"error: no .txt documents in {base / 'documents'}", file=sys.stderr)
        return 2
    out_dir = base / "documents_rewritten"

    budget = Budget(caps, args.budget_state)
    client = _MeteredClient(client_factory(arm), arm, budget)

    for src in sources:
        dest = out_dir / src.name
        if dest.exists() and not args.force:
            print(f"skip {src.name}: rewrite exists (use --force to redo)")
            continue
        try:
            text = rewrite_document(src.read_text(encoding="utf-8"), client)
        except BudgetExceeded as exc:
            print(f"stopped: budget exceeded before {src.name}: {exc}", file=sys.stderr)
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / "REVIEWED"
        if marker.exists():
            marker.unlink()
            print(f"WARNING: removed the stale marker {marker}. New unreviewed rewrites are "
                  f"being written, so you must re-review {out_dir} and create REVIEWED again.")
        dest.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {dest}")

    print(f"\nDone. Read each file in {out_dir} against its original in {base / 'documents'}, "
          f"edit where a fact changed, then create the file {out_dir / 'REVIEWED'} yourself. "
          f"This script never creates it, and level 2-3 trials refuse to run without it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
