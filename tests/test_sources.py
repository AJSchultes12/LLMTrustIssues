import json
from pathlib import Path

import pytest

from llmtrust.sources import (
    CutoffViolation,
    EvaluationLeak,
    UnreviewedRewrite,
    load_sources,
    select_context,
)
from llmtrust.types import Case, DecisionPoint, SourceDoc

CUTOFF = "2001-08-22"


def make_dp(cutoff=CUTOFF):
    return DecisionPoint(
        id="EN-3b", title="t", decision_date="2001-08-22", cutoff=cutoff,
        options=["a", "b"], actual_decision="x", edgar_forms=["10-Q"],
        email_custodians=["skilling"], role="r", scenario_brief="b", stake_statement="s",
    )


def make_case(dp=None, evaluation_only=None):
    dp = dp or make_dp()
    return Case(case_id="enron", company="Enron", decision_points=[dp],
                evaluation_only=evaluation_only or [], leakage_terms=[])


def raw_dir(root: Path, create=True) -> Path:
    d = root / "raw" / "enron" / "EN-3b"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_emails(root: Path, rows):
    write(raw_dir(root) / "emails.jsonl",
          "".join(json.dumps(r) + "\n" for r in rows))


def email(date_utc, subject="S", body="B"):
    return {"message_id": "m", "date_utc": date_utc, "from": "a@x", "to": "b@x",
            "cc": "", "subject": subject, "body": body}


def test_loads_all_kinds_before_cutoff(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "2001-08-14_enron_10-Q_0001.htm",
          "<html><head><style>p{}</style><script>var x=1;</script></head>"
          "<body><p>Revenue grew.</p><p>Net income fell.</p></body></html>")
    write(rd / "2001-06-01_enron_8-K_0002.txt", "plain filing text")
    write(rd / "manifest.json", json.dumps({"filings": [{"url": "https://sec/x"}]}))
    write_emails(tmp_path, [email("2001-08-21T23:59:00+00:00", "Subj", "Body text")])
    write(rd / "documents" / "2001-08-01_memo.txt", "memo text")
    docs = load_sources(make_case(), make_dp(), tmp_path)
    by_kind = {}
    for d in docs:
        by_kind.setdefault(d.kind, []).append(d)
    assert len(by_kind["filing"]) == 2 and len(by_kind["email"]) == 1
    htm = next(d for d in by_kind["filing"] if d.date == "2001-08-14")
    assert "Revenue grew." in htm.text and "<p>" not in htm.text and "var x" not in htm.text
    assert htm.title == "10-Q 2001-08-14"  # form and date only: no CIK or accession
    txt = next(d for d in by_kind["filing"] if d.date == "2001-06-01")
    assert txt.text == "plain filing text"
    assert by_kind["email"][0].title == "Subj" and by_kind["email"][0].date == "2001-08-21"
    assert "Body text" in by_kind["email"][0].text
    assert by_kind["document"][0].title == "memo"
    assert [d.date for d in docs] == sorted(d.date for d in docs)


SGML = """<SEC-DOCUMENT>0001024401-01-500010.txt : 20010814
<SEC-HEADER>0001024401-01-500010.hdr.sgml : 20010814
ACCESSION NUMBER:		0001024401-01-500010
CONFORMED SUBMISSION TYPE:	10-Q
CONFORMED PERIOD OF REPORT:	20010630
FILED AS OF DATE:		20010814
		CENTRAL INDEX KEY:			0001024401
		STANDARD INDUSTRIAL CLASSIFICATION:	NATURAL GAS TRANSMISSION [4922]
		IRS NUMBER:				470255140
		BUSINESS PHONE:		7138536161
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-Q
<TEXT>
Quarterly report body.
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def test_sgml_submission_header_is_stripped_and_title_has_no_accession(tmp_path):
    write(raw_dir(tmp_path) / "2001-08-14_enron_10-Q_0001024401-01-500010.txt", SGML)
    write(raw_dir(tmp_path) / "2001-05-15_enron_DEF 14A_0000950129-01-500183.htm",
          "<p>Proxy</p>")
    docs = load_sources(make_case(), make_dp(), tmp_path)
    q = next(d for d in docs if d.date == "2001-08-14")
    assert "Quarterly report body." in q.text
    for gone in ("20010814", "20010630", "0001024401", "470255140", "7138536161",
                 "ACCESSION", "CENTRAL INDEX KEY", "SEC-HEADER"):
        assert gone not in q.text
    assert q.title == "10-Q 2001-08-14"
    proxy = next(d for d in docs if d.date == "2001-05-15")
    assert proxy.title == "DEF 14A 2001-05-15"


def test_header_without_document_tag_is_still_stripped(tmp_path):
    text = "<SEC-HEADER>\nFILED AS OF DATE: 20010814\n</SEC-HEADER>\nBody."
    write(raw_dir(tmp_path) / "2001-08-14_enron_8-K_0001024401-01-500011.txt", text)
    (doc,) = load_sources(make_case(), make_dp(), tmp_path)
    assert "20010814" not in doc.text and "Body." in doc.text


def test_html_table_cells_are_separated(tmp_path):
    write(raw_dir(tmp_path) / "2001-08-14_enron_10-Q_0001.htm",
          "<table><tr><td>Kenneth L. Lay</td><td>Chairman</td></tr>"
          "<tr><th>$50,129</th><th>40,112</th></tr></table>"
          "<ul><li>one</li><li>two</li></ul>")
    (doc,) = load_sources(make_case(), make_dp(), tmp_path)
    lines = doc.text.splitlines()
    assert "Kenneth L. Lay\tChairman" in lines
    assert "$50,129\t40,112" in lines
    assert "one" in lines and "two" in lines
    assert "LayChairman" not in doc.text and "50,12940" not in doc.text


def test_late_document_rejected(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "documents" / "2001-08-23_x.txt", "late")
    with pytest.raises(CutoffViolation):
        load_sources(make_case(), make_dp(), tmp_path)


def test_document_on_cutoff_day_rejected(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "documents" / "2001-08-22_x.txt", "same day")
    with pytest.raises(CutoffViolation):
        load_sources(make_case(), make_dp(), tmp_path)


def test_late_filing_rejected(tmp_path):
    write(raw_dir(tmp_path) / "2001-08-22_enron_10-Q_0001.txt", "late")
    with pytest.raises(CutoffViolation):
        load_sources(make_case(), make_dp(), tmp_path)


def test_undated_file_fails_closed(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "documents" / "notes.txt", "no date")
    with pytest.raises(CutoffViolation):
        load_sources(make_case(), make_dp(), tmp_path)
    (rd / "documents" / "notes.txt").unlink()
    write(rd / "random_filing.htm", "no date")
    with pytest.raises(CutoffViolation):
        load_sources(make_case(), make_dp(), tmp_path)


def test_late_email_rejected(tmp_path):
    write_emails(tmp_path, [email("2001-08-21T10:00:00+00:00"),
                            email("2001-08-22T00:30:00+00:00")])
    with pytest.raises(CutoffViolation):
        load_sources(make_case(), make_dp(), tmp_path)


def test_evaluation_url_in_manifest_rejected(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "manifest.json", json.dumps(
        {"filings": [{"url": "https://sec/ok"}, {"url": "https://sec/bad"}]}))
    case = make_case(evaluation_only=[
        {"description": "d1", "url": None},
        {"description": "d2", "url": "https://sec/bad"},
    ])
    with pytest.raises(EvaluationLeak):
        load_sources(case, make_dp(), tmp_path)


def test_null_evaluation_url_does_not_match(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "manifest.json", json.dumps({"filings": [{"url": None}]}))
    case = make_case(evaluation_only=[{"description": "d", "url": None}])
    assert load_sources(case, make_dp(), tmp_path) == []


def test_missing_directory_returns_empty(tmp_path):
    assert load_sources(make_case(), make_dp(), tmp_path) == []


def test_level2_uses_reviewed_rewrite(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "documents" / "2001-08-01_memo.txt", "original")
    write(rd / "documents" / "2001-08-02_other.txt", "other original")
    write(rd / "documents_rewritten" / "2001-08-01_memo.txt", "rewritten")
    with pytest.raises(UnreviewedRewrite):
        load_sources(make_case(), make_dp(), tmp_path, level=2)
    # level 1 never touches the rewrite
    l1 = load_sources(make_case(), make_dp(), tmp_path, level=1)
    assert {d.text for d in l1} == {"original", "other original"}
    write(rd / "documents_rewritten" / "REVIEWED", "")
    # a partial rewrite set is refused even when marked REVIEWED: the original
    # famous memo must never reach a level 2-3 prompt
    with pytest.raises(UnreviewedRewrite, match="2001-08-02_other.txt"):
        load_sources(make_case(), make_dp(), tmp_path, level=2)
    write(rd / "documents_rewritten" / "2001-08-02_other.txt", "other rewritten")
    l2 = load_sources(make_case(), make_dp(), tmp_path, level=2)
    assert {d.text for d in l2} == {"rewritten", "other rewritten"}


@pytest.mark.parametrize("level", [2, 3])
def test_no_rewrite_dir_at_level_2_3_raises(tmp_path, level):
    write(raw_dir(tmp_path) / "documents" / "2001-08-01_memo.txt", "original")
    with pytest.raises(UnreviewedRewrite, match="2001-08-01_memo.txt"):
        load_sources(make_case(), make_dp(), tmp_path, level=level)


def test_full_rewrite_set_without_marker_raises(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "documents" / "2001-08-01_memo.txt", "original")
    write(rd / "documents_rewritten" / "2001-08-01_memo.txt", "rewritten")
    with pytest.raises(UnreviewedRewrite, match="REVIEWED"):
        load_sources(make_case(), make_dp(), tmp_path, level=3)


def test_level1_needs_no_rewrite(tmp_path):
    write(raw_dir(tmp_path) / "documents" / "2001-08-01_memo.txt", "original")
    assert [d.text for d in load_sources(make_case(), make_dp(), tmp_path, level=1)] == \
        ["original"]


def test_no_documents_needs_no_rewrite(tmp_path):
    write(raw_dir(tmp_path) / "2001-06-01_enron_8-K_0002.txt", "filing")
    assert len(load_sources(make_case(), make_dp(), tmp_path, level=2)) == 1


@pytest.mark.parametrize("where", ["documents", "twin"])
def test_non_txt_document_raises(tmp_path, where):
    if where == "documents":
        d = raw_dir(tmp_path) / "documents"
        level, scenario = 1, "real"
    else:
        d = tmp_path / "twin" / "enron" / "EN-3b" / "documents"
        level, scenario = 3, "twin"
    write(d / "2001-08-01_memo.txt", "memo")
    write(d / "2001-08-01_memo.pdf", "binary")
    with pytest.raises(ValueError, match="2001-08-01_memo.pdf"):
        load_sources(make_case(), make_dp(), tmp_path, level=level, scenario=scenario)


def test_rewrite_dir_with_late_file_is_not_a_back_door(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "documents" / "2001-08-01_memo.txt", "original")
    write(rd / "documents_rewritten" / "2001-08-01_memo.txt", "rewritten")
    write(rd / "documents_rewritten" / "REVIEWED", "")
    write(rd / "documents_rewritten" / "2001-09-01_extra.txt", "extra")
    docs = load_sources(make_case(), make_dp(), tmp_path, level=2)
    assert [d.text for d in docs] == ["rewritten"]


def test_twin_requires_level3_and_reads_twin_dir(tmp_path):
    rd = raw_dir(tmp_path)
    write(rd / "documents" / "2001-08-01_memo.txt", "real memo")
    write(rd / "2001-06-01_enron_8-K_0002.txt", "shared filing")
    write_emails(tmp_path, [email("2001-08-01T00:00:00+00:00", "Shared", "b")])
    twin = tmp_path / "twin" / "enron" / "EN-3b" / "documents"
    write(twin / "2001-08-05_twinmemo.txt", "twin memo")
    for level in (1, 2):
        with pytest.raises(ValueError):
            load_sources(make_case(), make_dp(), tmp_path, level=level, scenario="twin")
    docs = load_sources(make_case(), make_dp(), tmp_path, level=3, scenario="twin")
    texts = {d.text for d in docs if d.kind == "document"}
    assert texts == {"twin memo"}
    assert {d.kind for d in docs} == {"document", "filing", "email"}


def test_twin_documents_are_cutoff_checked(tmp_path):
    twin = tmp_path / "twin" / "enron" / "EN-3b" / "documents"
    write(twin / "2001-08-22_late.txt", "late")
    with pytest.raises(CutoffViolation):
        load_sources(make_case(), make_dp(), tmp_path, level=3, scenario="twin")


def test_unknown_scenario_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_sources(make_case(), make_dp(), tmp_path, scenario="other")


def D(date, kind, title, n):
    return SourceDoc(date=date, kind=kind, title=title, text="x" * n)


def test_select_context_keeps_documents_and_logs_drops():
    docs = [
        D("2001-08-01", "document", "memo", 500),
        D("2001-07-01", "filing", "f-old", 100),
        D("2001-08-10", "filing", "f-new", 100),
        D("2001-06-01", "email", "e1", 50),
        D("2001-07-15", "email", "e2", 50),
        D("2001-08-15", "email", "e3", 50),
    ]
    kept, log = select_context(docs, 10)  # far smaller than the memo alone
    assert [d.title for d in kept if d.kind == "document"] == ["memo"]
    assert next(d for d in kept if d.kind == "document").text == "x" * 500
    assert [d.kind for d in kept] == ["document"]
    dropped_titles = ["f-old", "f-new", "e1", "e2", "e3"]
    assert len(log) == len(dropped_titles)
    for title in dropped_titles:
        assert sum(title in entry for entry in log) == 1
    assert any("2001-06-01" in e and "e1" in e for e in log)


def test_select_context_emails_dropped_oldest_first():
    docs = [
        D("2001-08-01", "document", "memo", 100),
        D("2001-06-01", "email", "e-old", 100),
        D("2001-07-01", "email", "e-mid", 100),
        D("2001-08-01", "email", "e-new", 100),
    ]
    kept, log = select_context(docs, 250)  # memo 100 + 1 email fits... 2 do not
    titles = [d.title for d in kept]
    assert "memo" in titles and "e-new" in titles
    assert "e-old" not in titles and "e-mid" not in titles
    assert len(log) == 2 and any("e-old" in e for e in log) and any("e-mid" in e for e in log)


def test_select_context_truncates_filings_and_logs():
    docs = [
        D("2001-08-01", "filing", "big", 1000),
        D("2001-07-01", "filing", "small", 50),
    ]
    kept, log = select_context(docs, 400)  # per-filing cap 100
    big = next(d for d in kept if d.title == "big")
    assert len(big.text) == 100
    assert any("big" in e and "2001-08-01" in e for e in log)
    assert not any("small" in e for e in log)
    assert [d.date for d in kept] == sorted(d.date for d in kept)


def test_select_context_everything_fits_no_log():
    docs = [D("2001-08-01", "document", "m", 10), D("2001-07-01", "email", "e", 10)]
    kept, log = select_context(docs, 1000)
    assert kept == sorted(docs, key=lambda d: d.date) and log == []
