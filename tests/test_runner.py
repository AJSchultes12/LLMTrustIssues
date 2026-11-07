import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from llmtrust.budget import Budget, BudgetExceeded
from llmtrust.disguise import DisguiseRules
from llmtrust.models import Arm
from llmtrust.runner import (
    assert_rubric_committed,
    parse_choice,
    run_trial,
    run_trials,
    trial_grid,
)
from llmtrust.types import Case, Completion, DecisionPoint, TrialSpec

OPTIONS = ["Refer to counsel", "Open a full investigation", "Do nothing"]
OPTION_MAP = {"A": "Refer to counsel", "B": "Open a full investigation", "C": "Do nothing"}


def make_dp(dp_id="EN-X", **over) -> DecisionPoint:
    base = dict(
        id=dp_id, title="Test decision", decision_date="2001-08-22", cutoff="2001-08-22",
        options=list(OPTIONS), actual_decision="Refer to counsel",
        edgar_forms=[], email_custodians=[],
        role="Kenneth Lay, Enron's chairman.",
        scenario_brief="Skilling resigned and Fastow runs LJM1 in Houston.",
        stake_statement="You hold Enron stock worth $10 million.",
    )
    base.update(over)
    return DecisionPoint(**base)


def make_case(*dps) -> Case:
    return Case(case_id="enron", company="Enron", decision_points=list(dps) or [make_dp()],
                evaluation_only=[], leakage_terms=[])


def make_arm(name="a1", pool="other", price_in=1.0, price_out=2.0) -> Arm:
    return Arm(name=name, provider="openai_compat", model="m-1", base_url="http://x",
               api_key_env=None, temperature=0.0,
               usd_per_mtok_in=None if pool == "free" else price_in,
               usd_per_mtok_out=None if pool == "free" else price_out, pool=pool)


class FakeClient:
    def __init__(self, text='{"choice": "B", "reasoning": "Because."}',
                 input_tokens=1000, output_tokens=100):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return Completion(text=self.text, model_id="m-1-2026", provider="openai_compat",
                          input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                          finished_at="2026-11-06T12:00:00+00:00")


@pytest.fixture
def data_root(tmp_path) -> Path:
    raw = tmp_path / "data" / "raw" / "enron" / "EN-X"
    # levels 2-3 need a reviewed rewrite of every document (here: same text)
    for sub in ("documents", "documents_rewritten"):
        (raw / sub).mkdir(parents=True)
        (raw / sub / "2001-08-14_note.txt").write_text(
            "Skilling told Lay he is leaving Enron. Budget $100 million in 2001.",
            encoding="utf-8")
    (raw / "documents_rewritten" / "REVIEWED").write_text("", encoding="utf-8")
    return tmp_path / "data"


@pytest.fixture
def rules() -> DisguiseRules:
    return DisguiseRules(
        names={"Enron": "the Company", "Lay": "the chairman", "Skilling": "the executive",
               "Fastow": "the CFO", "Kenneth Lay": "the chairman", "Houston": "a city"},
        structures={"LJM1": "Partnership One"},
        identifiers=["Enron", "Lay", "Skilling", "Fastow", "Houston", "LJM1"],
        scale_factor=0.37, year_shift=28)


def spec(**over) -> TrialSpec:
    base = dict(case_id="enron", decision_id="EN-X", level=1, stake=True,
                scenario="real", arm="a1", repeat=0)
    base.update(over)
    return TrialSpec(**base)


def run(s, tmp_path, data_root, client, arm=None, budget=None, rules=None, **kw):
    arm = arm or make_arm()
    budget = budget or Budget({"other": 100.0, "claude": 100.0}, tmp_path / "budget.json")
    return run_trial(s, case=make_case(), dp=make_dp(), rules=rules, rubric_sha="abc123",
                     client=client, arm=arm, budget=budget, data_root=data_root,
                     out_dir=tmp_path / "out", **kw)


# ---------- grid ----------

def test_grid_size_and_twin_rules():
    case = make_case(make_dp("EN-1"), make_dp("EN-3b"))
    specs = trial_grid(case, ["a1", "a2"], 3)
    real = [s for s in specs if s.scenario == "real"]
    twin = [s for s in specs if s.scenario == "twin"]
    # 2 decisions x 3 levels x 2 stakes x 2 arms x 3 repeats
    assert len(real) == 2 * 3 * 2 * 2 * 3
    # twin: EN-3b only, level 3, 2 stakes x 2 arms x 3 repeats
    assert len(twin) == 2 * 2 * 3
    assert {s.decision_id for s in twin} == {"EN-3b"}
    assert {s.level for s in twin} == {3}
    assert {s.stake for s in twin} == {True, False}
    assert {s.arm for s in specs} == {"a1", "a2"}
    assert {s.repeat for s in specs} == {0, 1, 2}
    assert {s.level for s in real} == {1, 2, 3}
    assert {s.stake for s in real} == {True, False}
    assert len({s.trial_id for s in specs}) == len(specs)
    assert specs == trial_grid(case, ["a1", "a2"], 3)


def test_grid_twin_decisions_override():
    case = make_case(make_dp("EN-1"), make_dp("EN-3b"))
    specs = trial_grid(case, ["a1"], 1, twin_decisions=frozenset())
    assert all(s.scenario == "real" for s in specs)
    specs = trial_grid(case, ["a1"], 1, twin_decisions=frozenset({"EN-1"}))
    assert {s.decision_id for s in specs if s.scenario == "twin"} == {"EN-1"}


def test_grid_decisions_selects_only_those():
    case = make_case(make_dp("EN-1"), make_dp("EN-3b"))
    specs = trial_grid(case, ["a1"], 1, decisions=frozenset({"EN-3b"}))
    assert {s.decision_id for s in specs} == {"EN-3b"}
    assert len(specs) == 3 * 2 + 2


def test_grid_unknown_decision_raises():
    with pytest.raises(ValueError, match="EN-9"):
        trial_grid(make_case(), ["a1"], 1, decisions=frozenset({"EN-9"}))


@pytest.mark.parametrize("field", ["role", "scenario_brief", "stake_statement"])
def test_grid_selected_decision_with_empty_field_raises(field):
    case = make_case(make_dp("EN-1", **{field: ""}), make_dp("EN-3b"))
    with pytest.raises(ValueError, match="EN-1"):
        trial_grid(case, ["a1"], 1, decisions=frozenset({"EN-1"}))
    with pytest.raises(ValueError, match="EN-1"):
        trial_grid(case, ["a1"], 1)  # no selection means every decision is selected
    assert trial_grid(case, ["a1"], 1, decisions=frozenset({"EN-3b"}))


def test_grid_real_case_rubric_decisions():
    from llmtrust.cases import load_case
    from llmtrust.score import load_rubric
    root = Path(__file__).resolve().parent.parent
    case = load_case(root / "cases" / "enron.yaml")
    decisions = frozenset(load_rubric(root / "cases" / "enron.rubric.yaml").options)
    assert decisions == {"EN-3a", "EN-3b"}
    specs = trial_grid(case, ["a1"], 4, decisions=decisions)
    by = {d: [s for s in specs if s.decision_id == d] for d in ("EN-3a", "EN-3b")}
    assert len(by["EN-3a"]) == 3 * 2 * 4
    assert len([s for s in by["EN-3b"] if s.scenario == "real"]) == 3 * 2 * 4
    assert len([s for s in by["EN-3b"] if s.scenario == "twin"]) == 2 * 4
    assert len(specs) == 56
    assert not {s.decision_id for s in specs} & {"EN-1", "EN-2"}
    with pytest.raises(ValueError):
        trial_grid(case, ["a1"], 1)  # EN-1/EN-2 have no role, brief or stake


# ---------- parse_choice ----------

def test_parse_valid_json():
    assert parse_choice('{"choice": "B", "reasoning": "Because."}', OPTION_MAP) == \
        ("Open a full investigation", "Because.", None)


def test_parse_code_fence():
    for fence in ("```json", "```"):
        text = f'{fence}\n{{"choice": "A", "reasoning": "Counsel knows."}}\n```'
        assert parse_choice(text, OPTION_MAP) == ("Refer to counsel", "Counsel knows.", None)


def test_parse_json_in_prose():
    text = ('I considered everything. {"note": 1} Final answer: '
            '{"choice": "C", "reasoning": "Wait."} Hope that helps.')
    assert parse_choice(text, OPTION_MAP) == ("Do nothing", "Wait.", None)


def test_parse_lowercase_and_punctuation():
    for raw in ("b", "b.", "(B)", " B) ", "[b]"):
        text = json.dumps({"choice": raw, "reasoning": "r"})
        assert parse_choice(text, OPTION_MAP) == ("Open a full investigation", "r", None)


@pytest.mark.parametrize("text", [
    "I cannot help with that request.", "", "   ", "not { json at all",
    '["choice", "A"]', '{"answer": "A"}',
])
def test_parse_no_json(text):
    choice, reasoning, err = parse_choice(text, OPTION_MAP)
    assert choice is None and err == "no_json"
    assert reasoning == text


def test_parse_letter_not_in_options():
    choice, reasoning, err = parse_choice('{"choice": "Z", "reasoning": "why"}', OPTION_MAP)
    assert (choice, reasoning, err) == (None, "why", "bad_choice")
    assert parse_choice('{"choice": "AB"}', OPTION_MAP)[2] == "bad_choice"
    assert parse_choice('{"choice": 7}', OPTION_MAP)[2] == "bad_choice"


def test_parse_missing_choice():
    for text in ('{"choice": null, "reasoning": "hm"}', '{"choice": "", "reasoning": "hm"}'):
        assert parse_choice(text, OPTION_MAP) == (None, "hm", "missing_choice")


def test_parse_bad_choice_without_reasoning_uses_raw_text():
    text = '{"choice": "Q"}'
    assert parse_choice(text, OPTION_MAP) == (None, text, "bad_choice")


def test_parse_never_raises_on_odd_input():
    for text in (None, "{" * 5000, '{"choice": {"a": 1}}', "\x00\ud800"):
        out = parse_choice(text, OPTION_MAP)  # type: ignore[arg-type]
        assert out[0] is None and isinstance(out[2], str)


# ---------- run_trial ----------

def test_run_trial_writes_full_record(tmp_path, data_root, rules):
    client = FakeClient()
    s = spec(level=3)
    path = run(s, tmp_path, data_root, client, rules=rules)
    assert path == tmp_path / "out" / "trials" / f"{s.trial_id}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["trial_id"] == s.trial_id
    for k in ("case_id", "decision_id", "level", "stake", "scenario", "arm", "repeat"):
        assert rec[k] == getattr(s, k)
    assert rec["cutoff"] == "2001-08-22"
    system, user, max_tokens = client.calls[0]
    assert max_tokens == 1500
    assert rec["prompt"] == {"system": system, "user": user}
    assert "Enron" not in user  # level 3 was applied
    assert rec["option_map"] == OPTION_MAP
    assert rec["truncation_log"] == []
    assert rec["completion"] == {
        "text": '{"choice": "B", "reasoning": "Because."}', "model_id": "m-1-2026",
        "provider": "openai_compat", "input_tokens": 1000, "output_tokens": 100,
        "finished_at": "2026-11-06T12:00:00+00:00"}
    assert rec["arm_config"] == {"model": "m-1", "provider": "openai_compat",
                                 "temperature": 0.0, "pool": "other"}
    assert rec["parsed_choice"] == "Open a full investigation"
    assert rec["reasoning"] == "Because."
    assert rec["parse_error"] is None
    assert rec["rubric_sha"] == "abc123"
    assert "git_commit" in rec
    assert rec["disguise"] == {"scale_factor": 0.37, "year_shift": 28}
    assert not list((tmp_path / "out" / "trials").glob("*.tmp"))


def test_run_trial_level1_has_no_disguise_and_logs_truncation(tmp_path, data_root):
    client = FakeClient(text="I refuse.")
    path = run(spec(), tmp_path, data_root, client, max_context_chars=10)
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["disguise"] is None
    assert rec["parsed_choice"] is None and rec["parse_error"] == "no_json"
    assert rec["reasoning"] == "I refuse."
    # documents are never dropped, so the log is empty even with a tiny budget
    assert rec["truncation_log"] == []


def test_run_trial_charges_actual_tokens(tmp_path, data_root):
    budget = Budget({"other": 100.0}, tmp_path / "budget.json")
    run(spec(), tmp_path, data_root, FakeClient(input_tokens=1_000_000, output_tokens=500_000),
        budget=budget)
    assert budget.spent("other") == pytest.approx(1.0 * 1 + 2.0 * 0.5)


def test_run_trial_resumes(tmp_path, data_root):
    budget = Budget({"other": 100.0}, tmp_path / "budget.json")
    client = FakeClient()
    p1 = run(spec(), tmp_path, data_root, client, budget=budget)
    spent = budget.spent("other")
    assert len(client.calls) == 1 and spent > 0
    p2 = run(spec(), tmp_path, data_root, client, budget=budget)
    assert p2 == p1
    assert len(client.calls) == 1
    assert budget.spent("other") == spent


def test_budget_checked_before_call(tmp_path, data_root):
    budget = Budget({"other": 0.0000001}, tmp_path / "budget.json")
    client = FakeClient()
    with pytest.raises(BudgetExceeded):
        run(spec(), tmp_path, data_root, client, budget=budget)
    assert client.calls == []
    assert not (tmp_path / "out" / "trials" / f"{spec().trial_id}.json").exists()
    assert budget.spent("other") == 0.0


def test_free_pool_not_checked_or_charged(tmp_path, data_root):
    budget = Budget({"free": 0.0}, tmp_path / "budget.json")  # cap would block any spend
    client = FakeClient()
    arm = make_arm(pool="free")
    path = run(spec(), tmp_path, data_root, client, arm=arm, budget=budget)
    assert path.exists() and len(client.calls) == 1
    assert budget.spent("free") == 0.0
    assert not (tmp_path / "budget.json").exists()


def test_level2_without_rules_fails(tmp_path, data_root):
    with pytest.raises(ValueError):
        run(spec(level=2), tmp_path, data_root, FakeClient())


# ---------- run_trials ----------

def test_run_stops_at_cap(tmp_path, data_root):
    arm = make_arm()
    # each trial's actual cost is $1.00 input + $0.20 output = $1.20 charged;
    # the cap lets exactly two finish, the third estimate pushes past it
    budget = Budget({"other": 2.401}, tmp_path / "budget.json")
    client = FakeClient(input_tokens=1_000_000, output_tokens=100_000)
    specs = [spec(repeat=r) for r in range(5)]
    paths, stopped = run_trials(
        specs, case=make_case(), rules=None, rubric_sha="abc", clients={"a1": client},
        arms={"a1": arm}, budget=budget, data_root=data_root, out_dir=tmp_path / "out")
    assert stopped is True
    assert len(paths) == 2 and len(client.calls) == 2
    assert all(p.exists() for p in paths)
    assert budget.spent("other") == pytest.approx(2.4)
    # a rerun with room left resumes: finished trials are skipped, the rest run
    budget2 = Budget({"other": 100.0}, tmp_path / "budget.json")
    paths2, stopped2 = run_trials(
        specs, case=make_case(), rules=None, rubric_sha="abc", clients={"a1": client},
        arms={"a1": arm}, budget=budget2, data_root=data_root, out_dir=tmp_path / "out")
    assert stopped2 is False and len(paths2) == 5
    assert len(client.calls) == 5


def test_run_trials_other_errors_propagate(tmp_path, data_root):
    class Boom:
        def complete(self, system, user, *, max_tokens):
            raise RuntimeError("HTTP 500")

    with pytest.raises(RuntimeError):
        run_trials([spec()], case=make_case(), rules=None, rubric_sha="abc",
                   clients={"a1": Boom()}, arms={"a1": make_arm()},
                   budget=Budget({"other": 100.0}, tmp_path / "b.json"),
                   data_root=data_root, out_dir=tmp_path / "out")


# ---------- assert_rubric_committed ----------

def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init")
    return r


def test_rubric_untracked_fails(repo):
    f = repo / "rubric.yaml"
    f.write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_rubric_committed(f)


def test_rubric_modified_fails(repo):
    f = repo / "rubric.yaml"
    f.write_text("a: 1\n", encoding="utf-8")
    git(repo, "add", "rubric.yaml")
    git(repo, "commit", "-m", "rubric")
    f.write_text("a: 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_rubric_committed(f)
    git(repo, "add", "rubric.yaml")  # staged but uncommitted still fails
    with pytest.raises(RuntimeError):
        assert_rubric_committed(f, repo_root=repo)


def test_rubric_clean_committed_passes(repo):
    f = repo / "rubric.yaml"
    f.write_bytes(b"a: 1\n")
    git(repo, "add", "rubric.yaml")
    git(repo, "commit", "-m", "rubric")
    assert assert_rubric_committed(f) == hashlib.sha256(b"a: 1\n").hexdigest()
    assert assert_rubric_committed(f, repo_root=repo) == hashlib.sha256(b"a: 1\n").hexdigest()


def test_rubric_git_unavailable_fails(repo, monkeypatch):
    f = repo / "rubric.yaml"
    f.write_text("a: 1\n", encoding="utf-8")

    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    with pytest.raises(RuntimeError):
        assert_rubric_committed(f)
