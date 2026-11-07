from pathlib import Path

from llmtrust.cases import load_case

ROOT = Path(__file__).resolve().parent.parent


def test_enron_has_split_decisions():
    case = load_case(ROOT / "cases" / "enron.yaml")
    ids = [d.id for d in case.decision_points]
    assert "EN-3a" in ids and "EN-3b" in ids and "EN-3" not in ids
    dp = {d.id: d for d in case.decision_points}
    assert dp["EN-3a"].cutoff == "2001-08-14" and dp["EN-3b"].cutoff == "2001-08-22"
    assert all(d.role and d.scenario_brief and d.stake_statement for d in (dp["EN-3a"], dp["EN-3b"]))
    assert any(o for o in dp["EN-3a"].options if "uncertain" in o.lower())
    assert case.leakage_terms and "bankruptcy" in case.leakage_terms
