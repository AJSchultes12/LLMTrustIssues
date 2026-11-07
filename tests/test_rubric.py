from pathlib import Path

import pytest
import yaml

from llmtrust.cases import load_case
from llmtrust.score import OptionScore, Rubric, load_rubric

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ("EN-3a", "EN-3b")


def _rubric():
    return load_rubric(ROOT / "cases" / "enron.rubric.yaml")


def test_every_option_has_an_entry_keyed_by_exact_text():
    case = load_case(ROOT / "cases" / "enron.yaml")
    rubric = _rubric()
    dps = {d.id: d for d in case.decision_points}
    for did in DECISIONS:
        assert set(rubric.options[did]) == set(dps[did].options), did


def test_exactly_one_option_matches_executives_per_decision():
    rubric = _rubric()
    for did in DECISIONS:
        matches = [t for t, s in rubric.options[did].items() if s.matches_executives]
        assert len(matches) == 1, did


def test_alignment_in_range_and_types():
    rubric = _rubric()
    for did in DECISIONS:
        for s in rubric.options[did].values():
            assert isinstance(s, OptionScore)
            assert s.alignment in {0, 1, 2}
            assert isinstance(s.escalation, bool)


def test_executives_option_matches_actual_decision():
    # The option the executives took must not be rated as best-aligned.
    rubric = _rubric()
    for did in DECISIONS:
        for s in rubric.options[did].values():
            if s.matches_executives:
                assert s.alignment < 2


def test_reasoning_criteria_are_three_checks():
    assert len(_rubric().reasoning_criteria) == 3


def test_invalid_alignment_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "reasoning_criteria": ["a", "b", "c"],
        "decisions": {"X": {"opt": {"matches_executives": True, "alignment": 3, "escalation": False}}},
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rubric(bad)
