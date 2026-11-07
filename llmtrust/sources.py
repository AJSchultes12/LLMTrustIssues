"""Load source documents for a decision point and guard against hindsight leakage.

Every source must be dated strictly before the decision point's cutoff. A source
whose date cannot be determined is rejected too (fail closed). Nothing listed
under a case's ``evaluation_only`` may appear among the downloaded filings.
"""
from __future__ import annotations

import json
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from llmtrust.types import Case, DecisionPoint, SourceDoc


class CutoffViolation(Exception):
    """A source is dated on or after the cutoff, or has no parseable date."""


class EvaluationLeak(Exception):
    """A downloaded source is one of the case's evaluation-only items."""


class UnreviewedRewrite(Exception):
    """At level 2-3 a document has no rewrite, or the rewrites are not marked REVIEWED."""


_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.*)$", re.DOTALL)
_NON_FILING_NAMES = {"manifest.json", "emails.jsonl"}
_REVIEWED = "REVIEWED"
_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
               "table", "pre", "title", "hr"}
_CELL_TAGS = {"td", "th"}  # a tab between cells, so adjacent cells never glue together


# ---------- helpers ----------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in _CELL_TAGS:
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in _CELL_TAGS:
            self.parts.append("\t")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _strip_html(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    # collapse spaces within a cell; keep one tab between non-empty cells
    lines = ["\t".join(c for c in (" ".join(cell.split()) for cell in line.split("\t")) if c)
             for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _check_before_cutoff(source_date: str, cutoff: date, label: str) -> None:
    try:
        d = date.fromisoformat(source_date)
    except ValueError:
        raise CutoffViolation(f"{label}: unparseable date {source_date!r}") from None
    if d >= cutoff:
        raise CutoffViolation(f"{label}: dated {source_date}, not before cutoff {cutoff}")


def _dated_name(path: Path, cutoff: date, where: str) -> tuple[str, str]:
    """Return (iso_date, remainder of file name); raise if undated or too late."""
    m = _DATE_PREFIX.match(path.name)
    if not m:
        raise CutoffViolation(f"{where}/{path.name}: no YYYY-MM-DD_ date prefix")
    _check_before_cutoff(m.group(1), cutoff, f"{where}/{path.name}")
    return m.group(1), m.group(2)


def _urls(node) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "url" and isinstance(v, str):
                found.append(v)
            else:
                found.extend(_urls(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(_urls(item))
    return found


def _check_evaluation(case: Case, manifest: Path) -> None:
    banned = {e["url"] for e in case.evaluation_only if e.get("url")}
    if not banned or not manifest.is_file():
        return
    leaked = banned.intersection(_urls(json.loads(_read(manifest))))
    if leaked:
        raise EvaluationLeak(f"{manifest} lists evaluation-only url(s): {sorted(leaked)}")


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file())


def _load_filings(raw: Path, cutoff: date) -> list[SourceDoc]:
    docs = []
    for p in _files(raw):
        if p.name in _NON_FILING_NAMES:
            continue
        iso, _ = _dated_name(p, cutoff, str(raw))
        text = _read(p)
        if p.suffix.lower() in (".htm", ".html"):
            text = _strip_html(text)
        docs.append(SourceDoc(date=iso, kind="filing", title=p.name, text=text))
    return docs


def _load_emails(raw: Path, cutoff: date) -> list[SourceDoc]:
    path = raw / "emails.jsonl"
    if not path.is_file():
        return []
    docs = []
    for n, line in enumerate(_read(path).splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        day = str(rec.get("date_utc", ""))[:10]
        _check_before_cutoff(day, cutoff, f"{path}:{n}")
        subject = rec.get("subject", "")
        text = (f"From: {rec.get('from', '')}\nTo: {rec.get('to', '')}\n"
                f"Subject: {subject}\n\n{rec.get('body', '')}")
        docs.append(SourceDoc(date=day, kind="email", title=subject, text=text))
    return docs


def _document_files(docs_dir: Path) -> list[Path]:
    """Manual documents must be .txt, the only kind scripts/rewrite_memo.py rewrites."""
    files = _files(docs_dir)
    other = [p.name for p in files if p.suffix.lower() != ".txt"]
    if other:
        raise ValueError(f"{docs_dir} may hold only .txt documents; found {other}")
    return files


def _check_rewrites(docs_dir: Path, rewritten_dir: Path) -> None:
    """Level 2-3: every document needs a same-named rewrite and the REVIEWED marker."""
    originals = _document_files(docs_dir)
    if not originals:
        return
    missing = [p.name for p in originals if not (rewritten_dir / p.name).is_file()]
    if missing:
        raise UnreviewedRewrite(
            f"{rewritten_dir} lacks a rewrite of {missing}: levels 2-3 never use the "
            f"original documents (run scripts/rewrite_memo.py, review, create {_REVIEWED})")
    if not (rewritten_dir / _REVIEWED).is_file():
        raise UnreviewedRewrite(
            f"{rewritten_dir} has rewritten documents but no {_REVIEWED} marker")


def _load_documents(docs_dir: Path, cutoff: date,
                    rewritten_dir: Path | None) -> list[SourceDoc]:
    docs = []
    for p in _document_files(docs_dir):
        iso, slug = _dated_name(p, cutoff, str(docs_dir))
        source = p if rewritten_dir is None else rewritten_dir / p.name
        docs.append(SourceDoc(date=iso, kind="document",
                              title=Path(slug).stem, text=_read(source)))
    return docs


# ---------- public API ----------

def load_sources(case: Case, dp: DecisionPoint, data_root: Path, *,
                 level: int = 1, scenario: str = "real") -> list[SourceDoc]:
    if scenario not in ("real", "twin"):
        raise ValueError(f"unknown scenario {scenario!r}")
    if scenario == "twin" and level != 3:
        raise ValueError("the twin scenario is only defined at disguise level 3")
    data_root = Path(data_root)
    cutoff = date.fromisoformat(dp.cutoff)
    raw = data_root / "raw" / case.case_id / dp.id

    _check_evaluation(case, raw / "manifest.json")

    rewritten: Path | None = None
    if scenario == "twin":
        docs_dir = data_root / "twin" / case.case_id / dp.id / "documents"
    else:
        docs_dir = raw / "documents"
        if level >= 2:
            rewritten = raw / "documents_rewritten"
            _check_rewrites(docs_dir, rewritten)

    docs = (_load_filings(raw, cutoff) + _load_emails(raw, cutoff)
            + _load_documents(docs_dir, cutoff, rewritten))
    return sorted(docs, key=lambda d: d.date)


def _label(d: SourceDoc) -> str:
    return f"{d.kind} {d.date} {d.title!r}"


def select_context(docs: list[SourceDoc], max_chars: int) -> tuple[list[SourceDoc], list[str]]:
    """Fit docs into max_chars. Documents are never dropped; then filings (newest
    first, each cut to max_chars // 4), then emails (newest first, whole)."""
    order = {id(d): i for i, d in enumerate(docs)}
    log: list[str] = []
    kept: list[SourceDoc] = [d for d in docs if d.kind == "document"]
    budget = max_chars - sum(len(d.text) for d in kept)

    def newest_first(kind: str) -> list[SourceDoc]:
        return sorted((d for d in docs if d.kind == kind),
                      key=lambda d: (d.date, order[id(d)]), reverse=True)

    per_filing = max_chars // 4
    for kind in ("filing", "email"):
        out_of_room = False
        for d in newest_first(kind):
            item = d
            note = ""
            if kind == "filing" and len(d.text) > per_filing:
                item = SourceDoc(d.date, d.kind, d.title, d.text[:per_filing])
                note = f"truncated from {len(d.text)} to {per_filing} chars"
            if out_of_room or len(item.text) > budget:
                out_of_room = True
                log.append(f"dropped {_label(d)}: over the {max_chars}-char budget")
                continue
            budget -= len(item.text)
            kept.append(item)
            if note:
                log.append(f"{_label(d)}: {note}")
    kept.sort(key=lambda d: d.date)  # stable: ties keep documents, filings, emails order
    return kept, log
