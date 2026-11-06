"""Offline tests: cutoff filtering, EDGAR parsing, email parsing, case-file sanity."""
import io
import sys
import tarfile
from datetime import date
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_edgar import Filing, rows_from_block, select_filings  # noqa: E402
from filter_enron_emails import before_cutoff, custodian_of, parse_message  # noqa: E402

BLOCK = {
    "accessionNumber": ["0001-00-1", "0001-00-2", "0001-00-3", "0001-00-4"],
    "filingDate": ["1999-03-31", "1999-06-27", "1999-06-28", "1999-05-14"],
    "reportDate": ["1998-12-31", "", "", "1999-03-31"],
    "form": ["10-K", "8-K", "8-K", "10-Q"],
    "primaryDocument": ["", "a.htm", "b.htm", "q.txt"],
}


def test_rows_from_block_and_url_fallback():
    rows = rows_from_block(BLOCK, 1024401, "self")
    assert len(rows) == 4
    assert rows[0].url.endswith("/1024401/0001001/0001-00-1.txt")  # no primary doc -> full submission
    assert rows[1].url.endswith("/a.htm")


def test_cutoff_is_strict():
    rows = rows_from_block(BLOCK, 1024401, "self")
    kept = select_filings(rows, ["10-K", "10-Q", "8-K"], "1999-06-28")
    assert [f.filing_date for f in kept] == ["1999-03-31", "1999-05-14", "1999-06-27"]


def test_form_filter():
    rows = rows_from_block(BLOCK, 1024401, "self")
    assert [f.form for f in select_filings(rows, ["10-q"], "2000-01-01")] == ["10-Q"]


RAW = b"""Message-ID: <123.JavaMail.evans@thyme>
Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)
From: phillip.allen@enron.com
To: tim.belden@enron.com
Subject: Test

Here is the forecast.
"""


def test_parse_message_and_timezone():
    rec = parse_message(RAW)
    assert rec["date_utc"].startswith("2001-05-14T23:39")
    assert "forecast" in rec["body"]


def test_email_cutoff_uses_utc():
    # 16:39 PDT on May 14 is 23:39 UTC May 14 -> before a May 15 cutoff, not before May 14.
    rec = parse_message(RAW)
    assert before_cutoff(rec["date_utc"], "2001-05-15")
    assert not before_cutoff(rec["date_utc"], "2001-05-14")


def test_missing_date_rejected():
    assert parse_message(b"Subject: no date\n\nbody") is None


def test_custodian_parsing():
    assert custodian_of("maildir/lay-k/inbox/12.") == ("lay-k", "inbox")


def test_filter_script_end_to_end(tmp_path):
    import subprocess
    tar_path = tmp_path / "mini.tar.gz"
    late = RAW.replace(b"14 May 2001", b"30 Sep 2001").replace(b"<123", b"<456")
    with tarfile.open(tar_path, "w:gz") as t:
        for name, data in [("maildir/lay-k/inbox/1.", RAW),
                           ("maildir/lay-k/sent/1.", RAW),       # duplicate Message-ID
                           ("maildir/lay-k/inbox/2.", late)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    out = tmp_path / "out.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "filter_enron_emails.py"),
                    str(tar_path), "--cutoff", "2001-08-22", "--out", str(out)], check=True)
    lines = out.read_text().splitlines()
    assert len(lines) == 1  # late email excluded, duplicate removed


@pytest.mark.parametrize("case_file", sorted((ROOT / "cases").glob("*.yaml")))
def test_case_files_are_consistent(case_file):
    case = yaml.safe_load(case_file.read_text())
    ids = [dp["id"] for dp in case["decision_points"]]
    assert len(ids) == len(set(ids)), "duplicate decision ids"
    for dp in case["decision_points"]:
        cutoff = date.fromisoformat(dp["cutoff"])
        decided = date.fromisoformat(dp["decision_date"])
        assert cutoff <= decided, f"{dp['id']}: cutoff after decision date leaks information"
        assert dp["sources"]["edgar_forms"], f"{dp['id']}: no forms listed"
        assert len(dp["options_to_present"]) >= 2
        if "email_cutoff" in dp["sources"]:
            assert date.fromisoformat(dp["sources"]["email_cutoff"]) <= decided
    assert case["evaluation_only"], "every case needs ground truth kept out of prompts"
