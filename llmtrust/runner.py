"""Trial runner: build the grid, run one trial, resume, and write one JSON record each.

A trial record is the only thing scoring reads, so it carries the full prompt, the
raw reply, the exact model ID and provider, the rubric hash and the disguise
parameters. Records are written atomically and never overwritten: a trial whose
record exists is skipped without a model call or a charge.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from llmtrust.budget import Budget, BudgetExceeded, estimate_tokens
from llmtrust.disguise import DisguiseRules
from llmtrust.fileio import replace_with_retry
from llmtrust.models import Arm, ModelClient
from llmtrust.prompts import build_prompt
from llmtrust.sources import load_sources, select_context
from llmtrust.types import Case, Completion, DecisionPoint, TrialSpec

LEVELS = (1, 2, 3)
ESTIMATE_MARGIN = 1.2
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LETTER = re.compile(r"^[\W_]*([A-Za-z])[\W_]*$")


# ---------- grid ----------

def _selected_points(case: Case, decisions: frozenset[str] | None) -> list[DecisionPoint]:
    """The decision points to run; None means all. Each must have a role, brief and stake."""
    if decisions is not None:
        unknown = sorted(set(decisions) - {dp.id for dp in case.decision_points})
        if unknown:
            raise ValueError(f"decision id(s) {unknown} are not in case {case.case_id!r}")
    points = [dp for dp in case.decision_points if decisions is None or dp.id in decisions]
    for dp in points:
        empty = [f for f in ("role", "scenario_brief", "stake_statement")
                 if not getattr(dp, f).strip()]
        if empty:
            raise ValueError(f"decision {dp.id} has empty {', '.join(empty)}: it cannot be "
                             f"run (select other decisions)")
    return points


def trial_grid(case: Case, arms: list[str], repeats: int, *,
               decisions: frozenset[str] | None = None,
               twin_decisions: frozenset[str] = frozenset({"EN-3b"})) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for dp in _selected_points(case, decisions):
        for level in LEVELS:
            for stake in (True, False):
                for arm in arms:
                    for repeat in range(repeats):
                        specs.append(TrialSpec(case.case_id, dp.id, level, stake,
                                               "real", arm, repeat))
        if dp.id in twin_decisions:
            for stake in (True, False):
                for arm in arms:
                    for repeat in range(repeats):
                        specs.append(TrialSpec(case.case_id, dp.id, 3, stake,
                                               "twin", arm, repeat))
    return specs


# ---------- reply parsing ----------

def _find_choice_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict) and "choice" in obj:
            return obj
    return None


def parse_choice(text: str, option_map: dict[str, str]) -> tuple[str | None, str, str | None]:
    """Return (original option text or None, reasoning, error). Never raises."""
    try:
        raw = text if isinstance(text, str) else ("" if text is None else str(text))
        obj = _find_choice_object(raw)
        if obj is None:
            return None, raw, "no_json"
        reasoning = obj.get("reasoning")
        has_reasoning = isinstance(reasoning, str)
        fallback = reasoning if has_reasoning else raw
        choice = obj.get("choice")
        if choice is None or (isinstance(choice, str) and not choice.strip()):
            return None, fallback, "missing_choice"
        m = _LETTER.match(choice) if isinstance(choice, str) else None
        letter = m.group(1).upper() if m else None
        if letter is None or letter not in option_map:
            return None, fallback, "bad_choice"
        return option_map[letter], reasoning if has_reasoning else "", None
    except Exception:  # defensive: a malformed reply must never stop a batch
        return None, text if isinstance(text, str) else "", "no_json"


# ---------- git ----------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"git is unavailable: {exc}") from exc


def assert_rubric_committed(rubric_path: Path, *, repo_root: Path | None = None) -> str:
    """Return the SHA-256 of the rubric file, or raise if git does not hold it clean."""
    rubric_path = Path(rubric_path).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else rubric_path.parent
    tracked = _git(repo, "ls-files", "--error-unmatch", "--", str(rubric_path))
    if tracked.returncode != 0:
        raise RuntimeError(f"{rubric_path} is not tracked by git: commit the rubric before "
                           f"running trials ({tracked.stderr.strip()})")
    status = _git(repo, "status", "--porcelain", "--", str(rubric_path))
    if status.returncode != 0:
        raise RuntimeError(f"git status failed for {rubric_path}: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError(f"{rubric_path} has uncommitted changes: commit the rubric "
                           f"before running trials")
    return hashlib.sha256(rubric_path.read_bytes()).hexdigest()


def assert_rubric_approved(rubric_yaml: Path, rubric_md: Path) -> None:
    """Raise if either rubric file still carries the DRAFT marker (the user removes it
    when they approve the rubric)."""
    for path in (Path(rubric_yaml), Path(rubric_md)):
        if "DRAFT" in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"{path} is marked DRAFT: the rubric must be approved (and the "
                               f"DRAFT marker removed and committed) before any model run")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


# ---------- running ----------

def write_atomic(path: Path, record: dict) -> None:
    """Write a JSON record via a temp file and an atomic replace (retried on file locks)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def charged_complete(client: ModelClient, arm: Arm, budget: Budget, system: str, user: str,
                     max_tokens: int) -> Completion:
    """Check the budget (estimate with margin), call the model, record the actual cost.

    Pool "free" is neither checked nor charged. Raises BudgetExceeded before any call.
    When a priced arm's reply reports zero or missing usage, the pre-call estimate (with
    margin) is charged instead of $0 and the completion is marked cost_estimated.
    """
    if arm.pool == "free":
        return client.complete(system, user, max_tokens=max_tokens)

    estimate = budget.estimate_usd(
        arm, estimate_tokens(system + user), max_tokens) * ESTIMATE_MARGIN
    budget.check(arm.pool, estimate)

    completion = client.complete(system, user, max_tokens=max_tokens)

    if completion.input_tokens <= 0 or completion.output_tokens <= 0:
        budget.record(arm.pool, estimate)
        return replace(completion, cost_estimated=True)
    budget.record(arm.pool, budget.estimate_usd(
        arm, completion.input_tokens, completion.output_tokens))
    return completion


def run_trial(spec: TrialSpec, *, case: Case, dp: DecisionPoint, rules: DisguiseRules | None,
              rubric_sha: str, client: ModelClient, arm: Arm, budget: Budget,
              data_root: Path, out_dir: Path, max_context_chars: int = 120_000,
              max_tokens: int = 1500) -> Path:
    path = Path(out_dir) / "trials" / f"{spec.trial_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("rubric_sha")
        if existing != rubric_sha:
            raise RuntimeError(
                f"{spec.trial_id}: existing record was made under rubric {existing}, not the "
                f"current {rubric_sha}; use a fresh --out-dir instead of mixing rubrics")
        return path

    docs = load_sources(case, dp, Path(data_root), level=spec.level, scenario=spec.scenario)
    docs, truncation_log = select_context(docs, max_context_chars)
    prompt = build_prompt(case, dp, docs, level=spec.level, stake=spec.stake,
                          rules=rules, repeat=spec.repeat)

    completion = charged_complete(client, arm, budget, prompt.system, prompt.user, max_tokens)

    choice, reasoning, error = parse_choice(completion.text, prompt.option_map)
    record = {
        "trial_id": spec.trial_id,
        "case_id": spec.case_id,
        "decision_id": spec.decision_id,
        "level": spec.level,
        "stake": spec.stake,
        "scenario": spec.scenario,
        "arm": spec.arm,
        "repeat": spec.repeat,
        "cutoff": dp.cutoff,
        "prompt": {"system": prompt.system, "user": prompt.user},
        "option_map": prompt.option_map,
        "truncation_log": truncation_log,
        "completion": asdict(completion),
        "arm_config": {"model": arm.model, "provider": arm.provider,
                       "temperature": arm.temperature, "pool": arm.pool},
        "parsed_choice": choice,
        "reasoning": reasoning,
        "parse_error": error,
        "rubric_sha": rubric_sha,
        "git_commit": _git_commit(),
        "disguise": ({"scale_factor": rules.scale_factor, "year_shift": rules.year_shift}
                     if rules is not None and spec.level >= 2 else None),
    }
    write_atomic(path, record)
    return path


def run_trials(specs: list[TrialSpec], *, case: Case, rules: DisguiseRules | None,
               rubric_sha: str, clients: dict[str, ModelClient], arms: dict[str, Arm],
               budget: Budget, data_root: Path, out_dir: Path,
               **run_trial_kwargs) -> tuple[list[Path], bool]:
    """Run specs in order. Returns (record paths, stopped_by_budget)."""
    dps = {dp.id: dp for dp in case.decision_points}
    paths: list[Path] = []
    for spec in specs:
        try:
            paths.append(run_trial(
                spec, case=case, dp=dps[spec.decision_id], rules=rules,
                rubric_sha=rubric_sha, client=clients[spec.arm], arm=arms[spec.arm],
                budget=budget, data_root=data_root, out_dir=out_dir, **run_trial_kwargs))
        except BudgetExceeded:
            return paths, True
    return paths, False
