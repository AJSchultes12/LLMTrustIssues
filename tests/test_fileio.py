import os

import pytest

from llmtrust import fileio
from llmtrust.budget import Budget
from llmtrust.fileio import replace_with_retry
from llmtrust.runner import write_atomic


def flaky_replace(monkeypatch, failures):
    real = os.replace
    calls = []

    def fake(src, dst):
        calls.append((src, dst))
        if len(calls) <= failures:
            raise PermissionError(32, "The process cannot access the file")
        real(src, dst)

    monkeypatch.setattr(os, "replace", fake)
    monkeypatch.setattr(fileio.time, "sleep", lambda s: None)
    return calls


def test_replace_retries_then_succeeds(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "a.json"
    src.write_text("x", encoding="utf-8")
    calls = flaky_replace(monkeypatch, failures=2)
    replace_with_retry(src, dst)
    assert len(calls) == 3
    assert dst.read_text(encoding="utf-8") == "x" and not src.exists()


def test_replace_reraises_after_attempts(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "a.json"
    src.write_text("x", encoding="utf-8")
    calls = flaky_replace(monkeypatch, failures=99)
    with pytest.raises(PermissionError):
        replace_with_retry(src, dst, attempts=3)
    assert len(calls) == 3


def test_write_atomic_survives_transient_lock(tmp_path, monkeypatch):
    flaky_replace(monkeypatch, failures=2)
    write_atomic(tmp_path / "r.json", {"a": 1})
    assert (tmp_path / "r.json").read_text(encoding="utf-8").strip().startswith("{")
    assert not list(tmp_path.glob("*.tmp"))


def test_budget_save_survives_transient_lock(tmp_path, monkeypatch):
    calls = flaky_replace(monkeypatch, failures=2)
    Budget({"claude": 8.0}, tmp_path / "b.json").record("claude", 1.0)
    assert len(calls) == 3
    assert Budget({"claude": 8.0}, tmp_path / "b.json").spent("claude") == 1.0
