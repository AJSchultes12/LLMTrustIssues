#!/usr/bin/env python3
"""Command-line entry point for the experiment: estimate, probe, run, score.

    python scripts/run_experiment.py estimate --arms claude dev-free
    python scripts/run_experiment.py probe    --arms claude
    python scripts/run_experiment.py run      --arms claude dev-free
    python scripts/run_experiment.py score    --judge-arm claude

Exit codes: 0 success (also when a run stopped at a spend cap), 1 failure, 2 bad
setup (unknown arm, rubric not committed, probes missing).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmtrust.budget import Budget, BudgetExceeded, estimate_tokens  # noqa: E402
from llmtrust.cases import load_case  # noqa: E402
from llmtrust.disguise import DisguiseRules, load_rules  # noqa: E402
from llmtrust.models import Arm, ModelClient, load_arms, make_client  # noqa: E402
from llmtrust.probe import build_probe_prompt, probe_key, run_probe  # noqa: E402
from llmtrust.prompts import build_prompt  # noqa: E402
from llmtrust.runner import (  # noqa: E402
    ESTIMATE_MARGIN,
    assert_rubric_approved,
    assert_rubric_committed,
    charged_complete,
    run_trials,
    trial_grid,
    write_atomic,
)
from llmtrust.score import (  # noqa: E402
    judge_reasoning,
    load_rubric,
    population,
    score_trial,
    summarize,
    write_csv,
)
from llmtrust.sources import load_sources, select_context  # noqa: E402
from llmtrust.types import Case, DecisionPoint, Prompt, TrialSpec  # noqa: E402

# Same defaults as runner.run_trial and probe.run_probe; the estimate must match a real run.
MAX_CONTEXT_CHARS = 120_000
TRIAL_MAX_TOKENS = 1500
PROBE_MAX_TOKENS = 300
RUBRIC_DOC = ROOT / "docs" / "rubric.md"


class CliError(Exception):
    """Bad setup: reported to stderr, exit code 2."""


@dataclass
class Setup:
    case: Case
    dps: dict[str, DecisionPoint]
    arms: dict[str, Arm]
    caps: dict[str, float]
    rules: DisguiseRules | None
    out_dir: Path
    data_root: Path
    budget: Budget


class PromptBuilder:
    """Builds a trial's real prompt the way run_trial does, caching the loaded sources."""

    def __init__(self, setup: Setup):
        self.setup = setup
        self._docs: dict[tuple, list] = {}

    def build(self, spec: TrialSpec) -> Prompt:
        s = self.setup
        key = (spec.decision_id, spec.level, spec.scenario)
        if key not in self._docs:
            docs = load_sources(s.case, s.dps[spec.decision_id], s.data_root,
                                level=spec.level, scenario=spec.scenario)
            self._docs[key] = select_context(docs, MAX_CONTEXT_CHARS)[0]
        return build_prompt(s.case, s.dps[spec.decision_id], self._docs[key], level=spec.level,
                            stake=spec.stake, rules=s.rules, repeat=spec.repeat)


class ChargedClient:
    """Wraps a client so every call is budget-checked and charged (used by the judge)."""

    def __init__(self, client: ModelClient, arm: Arm, budget: Budget):
        self.client, self.arm, self.budget = client, arm, budget

    def complete(self, system: str, user: str, *, max_tokens: int):
        return charged_complete(self.client, self.arm, self.budget, system, user, max_tokens)


# ---------- setup ----------

def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--case", type=Path, default=ROOT / "cases" / "enron.yaml")
    common.add_argument("--rules", type=Path, default=ROOT / "cases" / "enron.disguise.yaml")
    common.add_argument("--rubric", type=Path, default=ROOT / "cases" / "enron.rubric.yaml")
    common.add_argument("--models", type=Path, default=ROOT / "config" / "models.yaml")
    common.add_argument("--data-root", type=Path, default=ROOT / "data")
    common.add_argument("--out-dir", type=Path, default=ROOT / "data" / "runs")
    common.add_argument("--budget-state", type=Path, default=None,
                        help="default: <out-dir>/budget.json")

    def with_arms(p: argparse.ArgumentParser) -> None:
        p.add_argument("--arms", nargs="+", required=True)
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--decisions", nargs="+", default=None,
                       help="decision ids to run; default: those with an entry in the rubric")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("estimate", "print trial counts and cost against the caps"),
                       ("probe", "run the recognition probes"),
                       ("run", "run the trials")):
        p = sub.add_parser(name, parents=[common], help=text)
        with_arms(p)
        if name == "run":
            p.add_argument("--skip-probes", action="store_true",
                           help="run without requiring probe records")
    p = sub.add_parser("score", parents=[common], help="score records and write summary.csv")
    p.add_argument("--judge-arm", default=None, help="arm that judges reasoning (costs money)")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def _setup(args: argparse.Namespace) -> Setup:
    case = load_case(args.case)
    arms = load_arms(args.models)
    for name in getattr(args, "arms", None) or []:
        if name not in arms:
            raise CliError(f"arm {name!r} is not in {args.models}; have {sorted(arms)}")
    caps = (yaml.safe_load(Path(args.models).read_text(encoding="utf-8")) or {}).get("caps") or {}
    rules = load_rules(args.rules) if Path(args.rules).is_file() else None
    state = args.budget_state or (args.out_dir / "budget.json")
    return Setup(case=case, dps={dp.id: dp for dp in case.decision_points}, arms=arms,
                 caps=caps, rules=rules, out_dir=Path(args.out_dir),
                 data_root=Path(args.data_root), budget=Budget(caps, state))


def selected_decisions(args: argparse.Namespace) -> frozenset[str]:
    """--decisions if given, else every decision id that has an entry in the rubric."""
    if getattr(args, "decisions", None):
        return frozenset(args.decisions)
    return frozenset(load_rubric(args.rubric).options)


def _grid(args: argparse.Namespace, s: Setup) -> list[TrialSpec]:
    try:
        return trial_grid(s.case, args.arms, args.repeats, decisions=selected_decisions(args))
    except ValueError as exc:
        raise CliError(str(exc)) from exc


def _trial_path(s: Setup, spec: TrialSpec) -> Path:
    return s.out_dir / "trials" / f"{spec.trial_id}.json"


def _probe_path(s: Setup, spec: TrialSpec) -> Path:
    return s.out_dir / "probes" / f"{probe_key(spec)}.json"


def _probe_specs(specs: list[TrialSpec]) -> dict[str, TrialSpec]:
    """First grid spec (repeat 0, first stake) for each probe key at level >= 2."""
    first: dict[str, TrialSpec] = {}
    for spec in specs:
        if spec.level >= 2:
            first.setdefault(probe_key(spec), spec)
    return first


# ---------- estimate ----------

def cmd_estimate(args, s: Setup, **_) -> int:
    specs = _grid(args, s)
    builder = PromptBuilder(s)
    probes = _probe_specs(specs)
    failed = False
    pool_usd: dict[str, float] = {}

    for name in args.arms:
        arm = s.arms[name]
        pending = [sp for sp in specs if sp.arm == name and not _trial_path(s, sp).exists()]
        in_tokens = 0
        for sp in pending:
            p = builder.build(sp)
            in_tokens += estimate_tokens(p.system + p.user)
        probe_in = 0
        needed = [sp for key, sp in probes.items() if sp.arm == name and not _probe_path(s, sp).exists()]
        for sp in needed:
            p = build_probe_prompt(builder.build(sp))
            probe_in += estimate_tokens(p.system + p.user)
        try:
            trial_usd = s.budget.estimate_usd(arm, in_tokens, len(pending) * TRIAL_MAX_TOKENS)
            probe_usd = s.budget.estimate_usd(arm, probe_in, len(needed) * PROBE_MAX_TOKENS)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed = True
            continue
        total = (trial_usd + probe_usd) * ESTIMATE_MARGIN
        pool_usd[arm.pool] = pool_usd.get(arm.pool, 0.0) + total
        print(f"{name}: {len(pending)} trials pending of "
              f"{sum(1 for sp in specs if sp.arm == name)}, {len(needed)} probes needed, "
              f"est ${total:.2f} (pool {arm.pool}, includes {ESTIMATE_MARGIN}x margin)")

    for pool, usd in sorted(pool_usd.items()):
        if pool == "free":
            print("pool free: $0.00 (no cap)")
            continue
        cap = s.caps.get(pool)
        if cap is None:
            print(f"error: pool {pool!r} has no cap: set caps.{pool} in {args.models}; "
                  f"run and probe refuse to spend from an uncapped pool", file=sys.stderr)
            failed = True
            continue
        over = "" if s.budget.spent(pool) + usd <= cap else "  OVER CAP"
        print(f"pool {pool}: est ${usd:.2f}, spent so far ${s.budget.spent(pool):.2f}, "
              f"cap ${cap:.2f}{over}")
    return 1 if failed else 0


# ---------- probe ----------

def cmd_probe(args, s: Setup, *, client_factory, **_) -> int:
    if s.rules is None:
        raise CliError(f"disguise rules file {args.rules} not found; probes need it")
    specs = _grid(args, s)
    builder = PromptBuilder(s)
    probes = _probe_specs(specs)
    for name in args.arms:
        arm = s.arms[name]
        client = client_factory(arm)
        done = 0
        for key, spec in probes.items():
            if spec.arm != name:
                continue
            if _probe_path(s, spec).exists():
                continue
            try:
                run_probe(spec, builder.build(spec), client=client, arm=arm, budget=s.budget,
                          rules=s.rules, out_dir=s.out_dir, max_tokens=PROBE_MAX_TOKENS)
                done += 1
            except BudgetExceeded as exc:
                print(f"{name}: stopped at the spend cap after {done} new probes: {exc}")
                break
        else:
            print(f"{name}: {done} new probes")
    return 0


# ---------- run ----------

def cmd_run(args, s: Setup, *, client_factory, rubric_checker, approval_checker, **_) -> int:
    try:
        rubric_sha = rubric_checker(args.rubric)
        rubric_checker(RUBRIC_DOC)
        approval_checker(args.rubric, RUBRIC_DOC)
    except Exception as exc:
        raise CliError(str(exc)) from exc

    specs = _grid(args, s)
    if not args.skip_probes:
        missing = [key for key, sp in _probe_specs(specs).items() if not _probe_path(s, sp).exists()]
        if missing:
            raise CliError("probe records are missing; run the probe command first (or pass "
                           "--skip-probes). Missing: " + ", ".join(missing))

    failed = False
    for name in args.arms:
        arm_specs = [sp for sp in specs if sp.arm == name]
        before = sum(1 for sp in arm_specs if _trial_path(s, sp).exists())
        try:
            paths, stopped = run_trials(
                arm_specs, case=s.case, rules=s.rules, rubric_sha=rubric_sha,
                clients={name: client_factory(s.arms[name])}, arms=s.arms, budget=s.budget,
                data_root=s.data_root, out_dir=s.out_dir)
        except Exception as exc:
            print(f"{name}: failed: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"{name}: {len(paths)}/{len(arm_specs)} records present, "
              f"{len(paths) - before} new"
              + (f"; stopped at the {s.arms[name].pool} pool cap" if stopped else ""))
    return 1 if failed else 0


# ---------- score ----------

def _load_json(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _judge_path(s: Setup, trial_id: str) -> Path:
    return s.out_dir / "judge" / f"{trial_id}.json"


def _cached_judgment(path: Path, rubric_sha: str, judge_arm: str) -> dict[str, bool] | None:
    """A judgment saved by an earlier score run, if it used the same rubric and judge."""
    if not path.is_file():
        return None
    cached = _load_json(path)
    if cached.get("rubric_sha") != rubric_sha or cached.get("judge_arm") != judge_arm:
        return None
    return cached["criteria"]


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def cmd_score(args, s: Setup, *, client_factory, rubric_checker, **_) -> int:
    try:
        rubric_checker(args.rubric)
    except Exception as exc:
        raise CliError(str(exc)) from exc
    current_sha = hashlib.sha256(Path(args.rubric).read_bytes()).hexdigest()
    rubric = load_rubric(args.rubric)
    paths = sorted((s.out_dir / "trials").glob("*.json"))
    records = [_load_json(p) for p in paths]
    stale = [r["trial_id"] for r in records if r.get("rubric_sha") != current_sha]
    if stale:
        raise CliError(f"{len(stale)} trial record(s) were made under a different rubric than "
                       f"the current {args.rubric} (sha {current_sha[:12]}), e.g. {stale[0]}; "
                       f"score them with that rubric or rerun into a fresh --out-dir")
    judge_client = None
    if args.judge_arm:
        if args.judge_arm not in s.arms:
            raise CliError(f"judge arm {args.judge_arm!r} is not in {args.models}")
        arm = s.arms[args.judge_arm]
        if arm.pool != "free" and s.caps.get(arm.pool) is None:
            raise CliError(f"judge arm {args.judge_arm!r}: pool {arm.pool!r} has no cap; "
                           f"set caps.{arm.pool} in {args.models}")
        judge_client = ChargedClient(client_factory(arm), arm, s.budget)

    scores = []
    judged_count: dict[str, list[int]] = {}   # trial arm -> [judged, eligible]
    for rec in records:
        spec = TrialSpec(rec["case_id"], rec["decision_id"], rec["level"], rec["stake"],
                         rec["scenario"], rec["arm"], rec["repeat"])
        probe_path = _probe_path(s, spec)
        probe = _load_json(probe_path) if rec["level"] >= 2 and probe_path.exists() else None
        judged = None
        if args.judge_arm and rec.get("parsed_choice") is not None:
            count = judged_count.setdefault(rec["arm"], [0, 0])
            count[1] += 1
            cache = _judge_path(s, rec["trial_id"])
            judged = _cached_judgment(cache, current_sha, args.judge_arm)
            if judged is None and judge_client is not None:
                try:
                    judged = judge_reasoning(judge_client, rubric, rec)
                    write_atomic(cache, {"trial_id": rec["trial_id"],
                                         "rubric_sha": current_sha,
                                         "judge_arm": args.judge_arm, "criteria": judged})
                except BudgetExceeded as exc:
                    print(f"judge stopped at the spend cap: {exc}", file=sys.stderr)
                    judge_client = None
                except Exception as exc:  # one bad reply or transient error: leave unjudged
                    print(f"{rec['trial_id']}: left unjudged: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
            if judged is not None:
                count[0] += 1
        scores.append(score_trial(rec, probe, rubric, judged, s.case.leakage_terms,
                                  rules=s.rules))
    for trial_arm, (n_judged, n_eligible) in sorted(judged_count.items()):
        print(f"judge: {trial_arm}: {n_judged}/{n_eligible} trials judged", file=sys.stderr)

    rows = summarize(scores)
    out = Path(args.results_dir) / "summary.csv"
    write_csv(rows, out)
    print(f"{len(scores)} trials scored; wrote {out}")
    unprobed = sum(1 for sc in scores if population(sc) == "unprobed")
    if unprobed:
        print(f"warning: {unprobed} unprobed level 2-3 trials (no probe record); they are "
              f"reported as population 'unprobed', not headline", file=sys.stderr)
    headline = [r for r in rows if r["population"] == "headline"]
    if headline:
        cols = ["arm", "decision_id", "level", "stake", "scenario", "population", "n",
                "n_invalid", "match_rate", "mean_alignment", "escalation_rate", "leak_rate",
                "mean_reasoning_score", "n_judged"]
        print("\t".join(cols))
        for r in headline:
            print("\t".join(_fmt(r[c]) for c in cols))
    return 0


# ---------- entry ----------

def main(argv: list[str] | None = None, *,
         client_factory: Callable[[Arm], ModelClient] | None = None,
         rubric_checker: Callable[[Path], str] | None = None,
         approval_checker: Callable[[Path, Path], None] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {"estimate": cmd_estimate, "probe": cmd_probe, "run": cmd_run, "score": cmd_score}
    try:
        setup = _setup(args)
        return handlers[args.command](
            args, setup, client_factory=client_factory or make_client,
            rubric_checker=rubric_checker or assert_rubric_committed,
            approval_checker=approval_checker or assert_rubric_approved)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
