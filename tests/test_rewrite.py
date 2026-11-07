import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from llmtrust.budget import estimate_tokens
from llmtrust.rewrite import REWRITE_SYSTEM, rewrite_document
from llmtrust.types import Completion
import rewrite_memo


class FakeClient:
    def __init__(self, reply="  rewritten text \n", in_tok=100, out_tok=50):
        self.reply, self.in_tok, self.out_tok = reply, in_tok, out_tok
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return Completion(text=self.reply, model_id="m", provider="p",
                          input_tokens=self.in_tok, output_tokens=self.out_tok,
                          finished_at="2026-11-06T00:00:00+00:00")


# ---------- rewrite_document ----------

def test_prompt_has_keep_facts_instruction_and_document():
    c = FakeClient()
    rewrite_document("Sherron wrote: $1.2 billion on 2001-08-15.", c)
    system, user, _ = c.calls[0]
    low = system.lower()
    assert "exactly as written" in low
    for word in ("number", "date", "name", "instrument"):
        assert word in low
    assert "15%" in system
    assert "only the rewritten text" in low
    assert user == "Sherron wrote: $1.2 billion on 2001-08-15."
    assert "Sherron" not in system


def test_returns_stripped_model_text():
    assert rewrite_document("x", FakeClient("\n\n  hello there \n")) == "hello there"


def test_default_max_tokens():
    c = FakeClient()
    short = "a" * 40
    rewrite_document(short, c)
    assert c.calls[0][2] == 2048
    long = "a" * 40000
    rewrite_document(long, c)
    assert c.calls[1][2] == 2 * estimate_tokens(long)


def test_explicit_max_tokens():
    c = FakeClient()
    rewrite_document("abc", c, max_tokens=77)
    assert c.calls[0][2] == 77


# ---------- CLI ----------

CASE = {
    "case_id": "enron", "company": "Enron Corp.",
    "decision_points": [{
        "id": "EN-3b", "title": "t", "decision_date": "2001-08-20",
        "cutoff": "2001-08-19", "options_to_present": ["a", "b"],
    }],
}


def _setup(tmp_path, cap=8.0, docs=("2001-08-15_memo.txt",)):
    case = tmp_path / "enron.yaml"
    case.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump({
        "caps": {"claude": cap},
        "arms": {"claude": dict(provider="anthropic", model="m", base_url=None,
                                api_key_env="NO_SUCH_KEY", temperature=0.0,
                                usd_per_mtok_in=1.0, usd_per_mtok_out=5.0,
                                pool="claude")},
    }), encoding="utf-8")
    data = tmp_path / "data"
    docs_dir = data / "raw" / "enron" / "EN-3b" / "documents"
    docs_dir.mkdir(parents=True)
    for name in docs:
        (docs_dir / name).write_text(f"original of {name}", encoding="utf-8")
    argv = ["--case", str(case), "--decision", "EN-3b", "--arm", "claude",
            "--data-root", str(data), "--models", str(models),
            "--budget-state", str(tmp_path / "budget.json")]
    return argv, data / "raw" / "enron" / "EN-3b"


def test_cli_writes_rewrites_and_never_reviewed(tmp_path, capsys):
    argv, base = _setup(tmp_path, docs=("2001-08-15_memo.txt", "2001-08-16_b.txt"))
    (base / "documents" / "ignore.md").write_text("not txt", encoding="utf-8")
    c = FakeClient("  new wording \n")
    rc = rewrite_memo.main(argv, client_factory=lambda arm: c)
    assert rc == 0
    out = base / "documents_rewritten"
    assert sorted(p.name for p in out.iterdir()) == ["2001-08-15_memo.txt", "2001-08-16_b.txt"]
    assert (out / "2001-08-15_memo.txt").read_text(encoding="utf-8") == "new wording\n"
    assert not (out / "REVIEWED").exists()
    assert "REVIEWED" in capsys.readouterr().out
    spent = json.loads((tmp_path / "budget.json").read_text())["spent"]["claude"]
    assert spent == pytest.approx(2 * (100 * 1.0 + 50 * 5.0) / 1e6)


def test_cli_skips_existing_without_force(tmp_path):
    argv, base = _setup(tmp_path)
    out = base / "documents_rewritten"
    out.mkdir()
    (out / "2001-08-15_memo.txt").write_text("hand edited", encoding="utf-8")
    c = FakeClient()
    assert rewrite_memo.main(argv, client_factory=lambda arm: c) == 0
    assert c.calls == []
    assert (out / "2001-08-15_memo.txt").read_text(encoding="utf-8") == "hand edited"
    assert rewrite_memo.main(argv + ["--force"], client_factory=lambda arm: c) == 0
    assert len(c.calls) == 1
    assert (out / "2001-08-15_memo.txt").read_text(encoding="utf-8") == "rewritten text\n"


def test_cli_stops_on_budget_exceeded(tmp_path, capsys):
    argv, base = _setup(tmp_path, cap=1e-9, docs=("2001-08-15_a.txt", "2001-08-16_b.txt"))
    c = FakeClient()
    rc = rewrite_memo.main(argv, client_factory=lambda arm: c)
    assert rc != 0
    assert c.calls == []
    out = base / "documents_rewritten"
    assert not out.exists() or list(out.iterdir()) == []
    assert "budget" in capsys.readouterr().err.lower()


def test_cli_budget_stops_after_first_document(tmp_path):
    # cap fits the first call's estimate (2048 out tokens ~ $0.0102) but not
    # the first call plus the second.
    argv, base = _setup(tmp_path, cap=0.012, docs=("2001-08-15_a.txt", "2001-08-16_b.txt"))
    c = FakeClient(in_tok=1000, out_tok=1000)  # actual $0.006
    rc = rewrite_memo.main(argv, client_factory=lambda arm: c)
    assert rc != 0
    assert len(c.calls) == 1
    assert [p.name for p in (base / "documents_rewritten").iterdir()] == ["2001-08-15_a.txt"]


# ---------- stale REVIEWED marker ----------

def _with_marker(base):
    out = base / "documents_rewritten"
    out.mkdir(exist_ok=True)
    (out / "REVIEWED").write_text("", encoding="utf-8")
    return out


def test_cli_removes_stale_reviewed_when_writing(tmp_path, capsys):
    argv, base = _setup(tmp_path)
    out = _with_marker(base)
    assert rewrite_memo.main(argv, client_factory=lambda arm: FakeClient()) == 0
    assert (out / "2001-08-15_memo.txt").exists()
    assert not (out / "REVIEWED").exists()
    captured = capsys.readouterr()
    assert "REVIEWED" in captured.out and "removed" in captured.out.lower()


def test_cli_keeps_reviewed_when_all_skipped(tmp_path):
    argv, base = _setup(tmp_path)
    out = _with_marker(base)
    (out / "2001-08-15_memo.txt").write_text("reviewed text", encoding="utf-8")
    c = FakeClient()
    assert rewrite_memo.main(argv, client_factory=lambda arm: c) == 0
    assert c.calls == []
    assert (out / "REVIEWED").exists()


def test_cli_keeps_reviewed_when_budget_stops_before_write(tmp_path):
    argv, base = _setup(tmp_path, cap=1e-9)
    out = _with_marker(base)
    assert rewrite_memo.main(argv, client_factory=lambda arm: FakeClient()) != 0
    assert (out / "REVIEWED").exists()
