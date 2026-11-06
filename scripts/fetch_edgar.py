#!/usr/bin/env python3
"""Download SEC EDGAR filings for each decision point, respecting its cutoff.

For every decision point in a case file, this keeps only filings whose filing
date is strictly before the cutoff, downloads them to
data/raw/<case>/<decision>/, and writes a manifest with SHA-256 hashes so the
dataset is reproducible.

SEC requires a descriptive User-Agent with contact info. Set it yourself:
    export SEC_USER_AGENT="flop-bench research your-name your-email@example.com"

Usage:
    python scripts/fetch_edgar.py cases/blockbuster.yaml
    python scripts/fetch_edgar.py cases/enron.yaml --decision EN-1 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import requests
import yaml

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
MIN_INTERVAL_S = 0.15  # SEC fair-access limit is 10 req/s; stay well under it.

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Filing:
    cik: int
    company_key: str
    form: str
    filing_date: str
    report_date: str
    accession: str
    primary_document: str

    @property
    def url(self) -> str:
        acc_nodash = self.accession.replace("-", "")
        # Older filings often have no primary document; fall back to the full submission text.
        doc = self.primary_document or f"{self.accession}.txt"
        return ARCHIVE_URL.format(cik=self.cik, acc_nodash=acc_nodash, doc=doc)


# ---------- pure logic (unit tested) ----------

def rows_from_block(block: dict, cik: int, company_key: str) -> list[Filing]:
    """Turn EDGAR's column-oriented filings block into Filing rows."""
    n = len(block.get("accessionNumber", []))
    get = lambda k, i: (block.get(k) or [""] * n)[i] or ""
    return [
        Filing(
            cik=cik,
            company_key=company_key,
            form=get("form", i),
            filing_date=get("filingDate", i),
            report_date=get("reportDate", i),
            accession=get("accessionNumber", i),
            primary_document=get("primaryDocument", i),
        )
        for i in range(n)
    ]


def select_filings(filings: list[Filing], forms: list[str], cutoff: str) -> list[Filing]:
    """Keep filings of the wanted forms filed strictly before the cutoff date."""
    cutoff_d = date.fromisoformat(cutoff)
    wanted = {f.upper() for f in forms}
    keep = [
        f for f in filings
        if f.form.upper() in wanted
        and f.filing_date
        and date.fromisoformat(f.filing_date) < cutoff_d
    ]
    return sorted(keep, key=lambda f: (f.filing_date, f.accession))


# ---------- network ----------

class Edgar:
    def __init__(self, user_agent: str):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0

    def get(self, url: str) -> requests.Response:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.s.get(url, timeout=60)
        self._last = time.monotonic()
        r.raise_for_status()
        return r

    def all_filings(self, cik: int, company_key: str) -> list[Filing]:
        root = self.get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json")).json()
        filings = rows_from_block(root["filings"]["recent"], cik, company_key)
        # Older history lives in paginated side files.
        for extra in root["filings"].get("files", []):
            page = self.get(SUBMISSIONS_URL.format(name=extra["name"])).json()
            filings += rows_from_block(page, cik, company_key)
        return filings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--decision", help="Only this decision id (e.g. BB-2)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--dry-run", action="store_true", help="List filings without downloading documents")
    args = ap.parse_args()

    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        print("Set SEC_USER_AGENT to a name and contact email (SEC requirement).", file=sys.stderr)
        return 2

    case = yaml.safe_load(args.case.read_text())
    edgar = Edgar(ua)
    companies = {"self": case["edgar"]["cik"], **case["edgar"].get("competitor_ciks", {})}
    cache: dict[str, list[Filing]] = {}

    for dp in case["decision_points"]:
        if args.decision and dp["id"] != args.decision:
            continue
        src = dp["sources"]
        keys = ["self", *src.get("include_competitor_filings", [])]
        chosen: list[Filing] = []
        for key in keys:
            if key not in cache:
                cache[key] = edgar.all_filings(companies[key], key)
            chosen += select_filings(cache[key], src["edgar_forms"], dp["cutoff"])

        out_dir = args.out / case["case_id"] / dp["id"]
        print(f"\n{dp['id']}  cutoff {dp['cutoff']}  -> {len(chosen)} filings")
        manifest = []
        for f in chosen:
            print(f"  {f.filing_date}  {f.company_key:8s} {f.form:8s} {f.accession}")
            entry = asdict(f) | {"url": f.url}
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                body = edgar.get(f.url).content
                name = f"{f.filing_date}_{f.company_key}_{f.form.replace('/', '-')}_{f.accession}{Path(f.url).suffix}"
                (out_dir / name).write_bytes(body)
                entry |= {"file": name, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            manifest.append(entry)

        if not args.dry_run:
            (out_dir / "manifest.json").write_text(json.dumps(
                {"case": case["case_id"], "decision": dp["id"], "cutoff": dp["cutoff"], "filings": manifest},
                indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
