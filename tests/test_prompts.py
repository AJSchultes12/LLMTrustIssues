from pathlib import Path

import pytest

from llmtrust.cases import load_case
from llmtrust.disguise import DisguiseRules, load_rules, residual_identifiers
from llmtrust.prompts import DisguiseLeak, build_prompt
from llmtrust.sources import CutoffViolation
from llmtrust.types import Case, DecisionPoint, SourceDoc

ROOT = Path(__file__).resolve().parent.parent


def make_dp(**over) -> DecisionPoint:
    base = dict(
        id="EN-X", title="Test decision", decision_date="2001-08-22", cutoff="2001-08-22",
        options=["Refer to counsel", "Open a full investigation", "Do nothing"],
        actual_decision="Refer to counsel",
        edgar_forms=[], email_custodians=[],
        role="Kenneth Lay, Enron's chairman.",
        scenario_brief="Skilling resigned and Fastow runs LJM1 in Houston.",
        stake_statement="You hold Enron stock worth $10 million.",
    )
    base.update(over)
    return DecisionPoint(**base)


def make_case(dp: DecisionPoint) -> Case:
    return Case(case_id="enron", company="Enron", decision_points=[dp],
                evaluation_only=[], leakage_terms=[])


DOCS = [
    SourceDoc("2001-08-14", "email", "Skilling resignation",
              "Skilling told Lay he is leaving Enron. Budget $100 million in 2001."),
    SourceDoc("2001-08-20", "document", "LJM1 note", "Raptors hedge LJM1 exposure."),
]


@pytest.fixture
def rules() -> DisguiseRules:
    return DisguiseRules(
        names={"Enron": "the Company", "Lay": "the chairman", "Skilling": "the executive",
               "Fastow": "the CFO", "Kenneth Lay": "the chairman", "Houston": "a city"},
        structures={"LJM1": "Partnership One", "Raptors": "Vehicles"},
        identifiers=["Enron", "Lay", "Skilling", "Fastow", "Houston", "LJM1", "Raptors"],
        scale_factor=0.37,
        year_shift=28,
    )


def build(dp=None, docs=DOCS, **kw):
    dp = dp or make_dp()
    kw.setdefault("level", 1)
    kw.setdefault("stake", False)
    kw.setdefault("rules", None)
    return build_prompt(make_case(dp), dp, docs, **kw)


def test_stake_statement_only_when_stake():
    with_stake = build(stake=True)
    without = build(stake=False)
    assert "You hold Enron stock" in with_stake.user
    assert "You hold Enron stock" not in without.user


def test_section_order():
    p = build(stake=True)
    u = p.user
    positions = [u.index(s) for s in (
        "Kenneth Lay, Enron's chairman.", "Skilling resigned", "You hold Enron stock",
        "Skilling resignation", "A. Refer to counsel", '{"choice"')]
    assert positions == sorted(positions)


def test_docs_rendered_with_date_kind_title_text():
    u = build().user
    assert "2001-08-14" in u and "email" in u and "Skilling resignation" in u
    assert "Skilling told Lay he is leaving Enron." in u


def test_brief_options_and_docs_all_disguised(rules):
    p = build(stake=True, level=3, rules=rules)
    assert residual_identifiers(p.user + p.system, rules) == []
    assert "Skilling" not in p.user and "Enron" not in p.user
    # dates and titles are disguised too: year shifted by 28
    assert "2029-08-14" in p.user
    assert "2001" not in p.user


def test_option_text_disguised_but_map_original(rules):
    dp = make_dp(options=["Ask Fastow to review LJM1", "Do nothing"])
    p = build(dp=dp, level=3, rules=rules)
    assert "A. Ask the CFO to review Partnership One" in p.user
    assert p.option_map["A"] == "Ask Fastow to review LJM1"


def test_level2_requires_rules():
    with pytest.raises(ValueError):
        build(level=2, rules=None)
    with pytest.raises(ValueError):
        build(level=3, rules=None)


def test_level1_leaves_text_unchanged_without_rules():
    p = build(level=1)
    assert "Enron" in p.user and "Skilling" in p.user


def test_option_order_rotates_with_repeat():
    dp = make_dp()
    n = len(dp.options)
    firsts = set()
    for r in range(n):
        p = build(dp=dp, repeat=r)
        assert sorted(p.option_map.values()) == sorted(dp.options)
        assert list(p.option_map) == ["A", "B", "C"]
        firsts.add(p.option_map["A"])
        assert f"A. {p.option_map['A']}" in p.user
    assert firsts == set(dp.options)
    assert build(dp=dp, repeat=0).option_map["A"] == dp.options[0]
    assert build(dp=dp, repeat=1).option_map["A"] == dp.options[1]
    assert build(dp=dp, repeat=n).option_map == build(dp=dp, repeat=0).option_map


def test_reply_format_instruction():
    p = build()
    assert '{"choice": "<letter>", "reasoning": "<text>"}' in p.user


def test_system_message_fixed_and_short():
    a, b = build(), build(stake=True, repeat=2)
    assert a.system == b.system
    assert "materials" in a.system


def test_no_source_on_or_after_cutoff():
    dp = make_dp()
    at_cutoff = [SourceDoc("2001-08-22", "email", "T", "x")]
    with pytest.raises(CutoffViolation):
        build(dp=dp, docs=at_cutoff)
    after = [DOCS[0], SourceDoc("2001-09-01", "email", "T", "x")]
    with pytest.raises(CutoffViolation):
        build(dp=dp, docs=after)


def test_brief_is_not_cutoff_checked():
    # a brief that mentions the cutoff day itself is fine; only docs are checked
    dp = make_dp(scenario_brief="On August 22, 2001 you met her.")
    assert "August 22, 2001" in build(dp=dp).user


def test_disguise_leak_raised(rules):
    leaky = DisguiseRules(
        names=rules.names, structures=rules.structures,
        identifiers=rules.identifiers + ["Nonremovable"],
        scale_factor=rules.scale_factor, year_shift=rules.year_shift,
    )
    docs = [SourceDoc("2001-08-14", "email", "T", "The Nonremovable thing.")]
    with pytest.raises(DisguiseLeak) as exc:
        build(docs=docs, level=2, rules=leaky)
    assert "Nonremovable" in str(exc.value)


@pytest.mark.parametrize("level", [2, 3])
def test_independent_audit_catches_replacer_blind_spot(rules, level):
    # "Enron" inside a compound word escapes the word-boundary replacer and
    # residual_identifiers; the substring audit must still stop the prompt
    docs = [SourceDoc("2001-08-14", "email", "T", "See EnronBroadband for details.")]
    with pytest.raises(DisguiseLeak) as exc:
        build(docs=docs, level=level, rules=rules)
    assert "Enron" in str(exc.value)
    docs = [SourceDoc("2001-08-14", "email", "T", "From: lay@corp.example")]
    with pytest.raises(DisguiseLeak):
        build(docs=docs, level=level, rules=rules)


def test_real_case_end_to_end_level3():
    case = load_case(ROOT / "cases" / "enron.yaml")
    rules = load_rules(ROOT / "cases" / "enron.disguise.yaml")
    dp = next(d for d in case.decision_points if d.id == "EN-3b")
    p = build_prompt(case, dp, [], level=3, stake=True, rules=rules)
    assert residual_identifiers(p.user + p.system, rules, 3) == []
    assert "Enron" not in p.user and "Lay" not in p.user
    assert set(p.option_map.values()) == set(dp.options)


@pytest.mark.parametrize("dp_id", ["EN-3a", "EN-3b"])
@pytest.mark.parametrize("level", [2, 3])
def test_real_case_briefs_pass_independent_audit(dp_id, level):
    from llmtrust.disguise import audit_text
    case = load_case(ROOT / "cases" / "enron.yaml")
    rules = load_rules(ROOT / "cases" / "enron.disguise.yaml")
    dp = next(d for d in case.decision_points if d.id == dp_id)
    for repeat in range(len(dp.options)):
        p = build_prompt(case, dp, [], level=level, stake=True, rules=rules, repeat=repeat)
        assert audit_text(p.system + "\n" + p.user, rules, level) == []
