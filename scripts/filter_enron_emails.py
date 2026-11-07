#!/usr/bin/env python3
"""Filter the CMU Enron email corpus down to messages sent before a cutoff.

Streams the tarball (no need to extract 0.5M files to disk), keeps messages
dated strictly before the cutoff, de-duplicates by Message-ID (the same email
appears in many folders), and writes JSON Lines.

Download the corpus first (about 1.7 GB):
    curl -O https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

Usage:
    python scripts/filter_enron_emails.py enron_mail_20150507.tar.gz \
        --cutoff 1999-06-27 --out data/raw/enron/EN-1/emails.jsonl
    # optionally restrict to specific custodians (maildir folder names):
    python scripts/filter_enron_emails.py corpus.tar.gz --cutoff 2001-08-14 \
        --custodians lay-k skilling-j --out data/raw/enron/EN-3a/emails.jsonl
    python scripts/filter_enron_emails.py corpus.tar.gz --cutoff 2001-08-22 \
        --custodians lay-k skilling-j --out data/raw/enron/EN-3b/emails.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

_parser = BytesParser(policy=policy.compat32)


def parse_message(raw: bytes) -> dict | None:
    """Parse one maildir file. Returns None if it has no usable Date header."""
    msg = _parser.parsebytes(raw)
    try:
        dt = parsedate_to_datetime(msg.get("Date", ""))
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    payload = msg.get_payload(decode=True)
    body = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else str(msg.get_payload())
    return {
        "message_id": (msg.get("Message-ID") or "").strip(),
        "date_utc": dt.astimezone(timezone.utc).isoformat(),
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "cc": msg.get("Cc", ""),
        "subject": msg.get("Subject", ""),
        "body": body,
    }


def before_cutoff(date_utc_iso: str, cutoff: str) -> bool:
    """True if the message date (UTC) falls strictly before the cutoff day (UTC midnight)."""
    cutoff_dt = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(date_utc_iso) < cutoff_dt


def custodian_of(member_name: str) -> tuple[str, str]:
    """maildir/<custodian>/<folder...>/<n>. -> (custodian, folder path)."""
    parts = member_name.split("/")
    if "maildir" in parts:
        parts = parts[parts.index("maildir") + 1:]
    custodian = parts[0] if parts else ""
    folder = "/".join(parts[1:-1])
    return custodian, folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tarball", type=Path)
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD; keep messages strictly before this day (UTC)")
    ap.add_argument("--custodians", nargs="*", help="Only these maildir custodians (e.g. lay-k skilling-j)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wanted = set(args.custodians or [])
    seen: set[str] = set()
    kept = scanned = no_date = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.tarball, "r|gz") as tar, args.out.open("w", encoding="utf-8") as out:
        for member in tar:
            if not member.isfile():
                continue
            custodian, folder = custodian_of(member.name)
            if wanted and custodian not in wanted:
                continue
            scanned += 1
            fh = tar.extractfile(member)
            if fh is None:
                continue
            rec = parse_message(fh.read())
            if rec is None:
                no_date += 1
                continue
            if not before_cutoff(rec["date_utc"], args.cutoff):
                continue
            key = rec["message_id"] or f"{custodian}/{member.name}"
            if key in seen:
                continue
            seen.add(key)
            rec |= {"custodian": custodian, "folder": folder}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if scanned % 50000 == 0:
                print(f"  scanned {scanned:,}  kept {kept:,}", file=sys.stderr)

    print(f"Scanned {scanned:,} messages, kept {kept:,} unique before {args.cutoff} "
          f"({no_date:,} skipped for missing/invalid dates) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
