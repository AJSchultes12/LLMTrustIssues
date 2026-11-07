import json

import pytest

from llmtrust.budget import Budget, BudgetExceeded
from llmtrust.disguise import DisguiseRules
from llmtrust.models import Arm
from llmtrust.probe import build_probe_prompt, is_identified, probe_key, run_probe
from llmtrust.prompts import OPTIONS_HEADER, REPLY_FORMAT, SYSTEM_PROMPT, build_prompt
from llmtrust.types import Case, Completion, DecisionPoint, Prompt, SourceDoc, TrialSpec

RULES = DisguiseRules(
    names={"Enron": "the Company", "Lay": "the chairman", "Skilling": "the executive",
           "Houston": "a city"},
    structures={"LJM1": "Partnership One", "Raptor": "Vehicle Two"},
    identifiers=["Enron", "Lay", "Skilling", "Houston", "LJM1", "Raptor"],
    scale_factor=0.37, year_shift=28)

OPTIONS = ["Refer to counsel", "Open a full investigation", "Do nothing"]


def make_dp() -> DecisionPoint:
    return DecisionPoint(
        id="EN-X", title="Test decision", decision_date="2001-08-22", cutoff="2001-08-22",
        options=list(OPTIONS), actual_decision="Refer to counsel",
        edgar_forms=[], email_custodians=[],
        role="Kenneth Lay, Enron's chairman.",
        scenario_brief="Skilling resigned and runs LJM1 in Houston.",
        stake_statement="You hold Enron stock worth $10 million.")


def make_prompt(level=2):
    dp = make_dp()
    case = Case(case_id="enron", company="Enron", decision_points=[dp],
                evaluation_only=[], leakage_terms=[])
    docs = [SourceDoc(date="2001-08-14", kind="document", title="Note",
                      text="He is leaving. Budget $100 million in 2001.")]
    return build_prompt(case, dp, docs, level=level, stake=True, rules=RULES)


def spec(**over) -> TrialSpec:
    base = dict(case_id="enron", decision_id="EN-X", level=2, stake=True,
                scenario="real", arm="a1", repeat=0)
    base.update(over)
    return TrialSpec(**base)


def make_arm(pool="other") -> Arm:
    return Arm(name="a1", provider="openai_compat", model="m-1", base_url="http://x",
               api_key_env=None, temperature=0.0,
               usd_per_mtok_in=None if pool == "free" else 1.0,
               usd_per_mtok_out=None if pool == "free" else 2.0, pool=pool)


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return Completion(text=self.text, model_id="m-1-2026", provider="openai_compat",
                          input_tokens=1000, output_tokens=100,
                          finished_at="2026-11-06T12:00:00+00:00")


def budget(tmp_path):
    return Budget({"claude": 8.0, "other": 20.0}, tmp_path / "budget.json")


# ---------- probe_key ----------

def test_probe_key_ignores_stake_and_repeat_but_includes_the_rest():
    base = probe_key(spec())
    assert base == "enron.EN-X.L2.real.a1"
    assert probe_key(spec(stake=False)) == base
    assert probe_key(spec(repeat=3)) == base
    assert probe_key(spec(level=3)) != base
    assert probe_key(spec(scenario="twin")) != base
    assert probe_key(spec(arm="a2")) != base
    assert probe_key(spec(decision_id="EN-Y")) != base


# ---------- probe prompt ----------

def test_probe_prompt_asks_company_year_ending_and_drops_the_question():
    prompt = make_prompt()
    probe = build_probe_prompt(prompt)
    assert probe.system == "You are a careful business historian."
    assert probe.option_map == {}
    # keeps the materials, role and stake text
    assert "Materials:" in probe.user
    assert "He is leaving." in probe.user
    assert "Your role:" in probe.user
    # asks for company, year, ending in JSON
    low = probe.user.lower()
    assert "company" in low and "year" in low and "ended" in low
    assert '{"company": "...", "year": "...", "outcome": "..."}' in probe.user
    # none of the decision question
    for opt in OPTIONS:
        assert opt not in probe.user
    assert OPTIONS_HEADER not in probe.user
    assert REPLY_FORMAT not in probe.user
    assert '"choice"' not in probe.user
    assert SYSTEM_PROMPT not in probe.user


def test_probe_prompt_requires_the_options_header():
    with pytest.raises(ValueError):
        build_probe_prompt(Prompt(system="s", user="no options here", option_map={}))


# ---------- is_identified ----------

def test_is_identified_matches_long_keys_case_insensitively():
    assert is_identified("I think this is ENRON in 2001.", RULES)
    assert is_identified("It resembles enron's collapse", RULES)
    assert is_identified("A company in houston", RULES)


def test_is_identified_short_keys_are_case_sensitive():
    assert is_identified("The chairman, Lay, resigned", RULES)
    assert not is_identified("They decided to lay off staff; a relay failed", RULES)


def test_is_identified_false_for_generic_description():
    assert not is_identified(
        '{"company": "a large energy trader", "year": "1999", "outcome": "collapse"}', RULES)


def test_is_identified_flags_structure_names_only_at_level_3():
    answer = "This looks like LJM1 and Raptor"
    assert is_identified(answer, RULES, 3)
    assert not is_identified(answer, RULES, 2)


def test_is_identified_default_level_is_3():
    assert is_identified("This looks like LJM1", RULES)
    assert is_identified("Enron", RULES) and is_identified("Enron", RULES, 2)


# ---------- run_probe ----------

NAMES_ENRON = '{"company": "Enron", "year": "2001", "outcome": "bankruptcy"}'
GENERIC = '{"company": "an energy trading firm", "year": "unknown", "outcome": "unclear"}'


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_probe_writes_record_and_flags_identified(tmp_path):
    client = FakeClient(NAMES_ENRON)
    prompt = make_prompt()
    path = run_probe(spec(), prompt, client=client, arm=make_arm(), budget=budget(tmp_path),
                     rules=RULES, out_dir=tmp_path)
    assert path == tmp_path / "probes" / "enron.EN-X.L2.real.a1.json"
    rec = read(path)
    assert rec["identified"] is True
    assert rec["probe_key"] == "enron.EN-X.L2.real.a1"
    assert (rec["case_id"], rec["decision_id"], rec["level"], rec["scenario"], rec["arm"]) == \
        ("enron", "EN-X", 2, "real", "a1")
    assert rec["prompt"]["system"] == "You are a careful business historian."
    assert rec["prompt"]["user"] == build_probe_prompt(prompt).user
    c = rec["completion"]
    assert c["text"] == NAMES_ENRON and c["model_id"] == "m-1-2026"
    assert c["provider"] == "openai_compat"
    assert (c["input_tokens"], c["output_tokens"]) == (1000, 100)
    assert c["finished_at"] == "2026-11-06T12:00:00+00:00"
    assert rec["finished"] == "2026-11-06"
    assert client.calls[0][2] == 300
    assert not list((tmp_path / "probes").glob("*.tmp"))


def test_run_probe_generic_answer_not_identified(tmp_path):
    path = run_probe(spec(level=3), make_prompt(3), client=FakeClient(GENERIC),
                     arm=make_arm(), budget=budget(tmp_path), rules=RULES, out_dir=tmp_path)
    assert read(path)["identified"] is False


def test_run_probe_resumes_without_call_or_charge(tmp_path):
    b = budget(tmp_path)
    p1 = run_probe(spec(), make_prompt(), client=FakeClient(GENERIC), arm=make_arm(),
                   budget=b, rules=RULES, out_dir=tmp_path)
    spent = b.spent("other")
    second = FakeClient(NAMES_ENRON)
    p2 = run_probe(spec(stake=False, repeat=2), make_prompt(), client=second, arm=make_arm(),
                   budget=b, rules=RULES, out_dir=tmp_path)
    assert p2 == p1
    assert second.calls == []
    assert b.spent("other") == spent
    assert read(p2)["identified"] is False


def test_run_probe_charges_priced_pool_and_not_free_pool(tmp_path):
    b = budget(tmp_path)
    run_probe(spec(), make_prompt(), client=FakeClient(GENERIC), arm=make_arm("other"),
              budget=b, rules=RULES, out_dir=tmp_path)
    charged = (1000 * 1.0 + 100 * 2.0) / 1_000_000
    assert b.spent("other") == pytest.approx(charged)

    run_probe(spec(scenario="twin", level=3), make_prompt(3), client=FakeClient(GENERIC),
              arm=make_arm("free"), budget=b, rules=RULES, out_dir=tmp_path)
    assert b.spent("free") == 0.0
    assert b.spent("other") == pytest.approx(charged)


def test_run_probe_stops_at_cap_before_calling(tmp_path):
    b = Budget({"other": 0.0000001}, tmp_path / "b.json")
    client = FakeClient(GENERIC)
    with pytest.raises(BudgetExceeded):
        run_probe(spec(), make_prompt(), client=client, arm=make_arm(), budget=b,
                  rules=RULES, out_dir=tmp_path)
    assert client.calls == []
    assert not (tmp_path / "probes").exists()


def test_run_probe_rejects_level_1(tmp_path):
    client = FakeClient(GENERIC)
    with pytest.raises(ValueError):
        run_probe(spec(level=1), make_prompt(1), client=client, arm=make_arm(),
                  budget=budget(tmp_path), rules=RULES, out_dir=tmp_path)
    assert client.calls == []


def test_run_probe_uses_spec_level_for_structure_names(tmp_path):
    reply = '{"company": "LJM1 and Raptor", "year": "unknown", "outcome": "unclear"}'
    l3 = run_probe(spec(level=3), make_prompt(3), client=FakeClient(reply), arm=make_arm(),
                   budget=budget(tmp_path), rules=RULES, out_dir=tmp_path)
    l2 = run_probe(spec(level=2), make_prompt(2), client=FakeClient(reply), arm=make_arm(),
                   budget=budget(tmp_path), rules=RULES, out_dir=tmp_path)
    assert read(l3)["identified"] is True
    assert read(l2)["identified"] is False
