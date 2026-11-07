"""Disguise levels 1-3: hide the identity of a case while keeping the decision facts.

Level 1 returns text unchanged. Level 2 replaces real names (company, people,
products, places). Level 3 also renames structures, scales every dollar amount by
one fixed factor and shifts every year by a fixed number of years.

A year shift of 28 is used because the Gregorian calendar repeats every 28 years
between 1901 and 2099, so weekdays and day gaps between dates survive the shift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml

# Keys of this length or shorter match case-sensitively ("Lay" must not hit "lay off").
_SHORT_KEY_MAX = 4
_LEFT = r"(?<![A-Za-z0-9])"
_RIGHT = r"(?![A-Za-z0-9])"


@dataclass(frozen=True)
class DisguiseRules:
    names: dict[str, str]
    structures: dict[str, str]
    identifiers: list[str]
    scale_factor: float
    year_shift: int


def load_rules(path: Path) -> DisguiseRules:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DisguiseRules(
        names={str(k): str(v) for k, v in (raw.get("names") or {}).items()},
        structures={str(k): str(v) for k, v in (raw.get("structures") or {}).items()},
        identifiers=[str(x) for x in (raw.get("identifiers") or [])],
        scale_factor=float(raw["scale_factor"]),
        year_shift=int(raw["year_shift"]),
    )


# ---------- term replacement ----------

def _key_regex(key: str) -> str:
    body = re.escape(key)
    return body if len(key) <= _SHORT_KEY_MAX else f"(?i:{body})"


def _term_pattern(keys: list[str]) -> re.Pattern[str]:
    """One alternation, longest key first, so "Enron Corp." beats "Enron"."""
    ordered = sorted(keys, key=len, reverse=True)
    alts = "|".join(f"({_key_regex(k)})" for k in ordered)
    return re.compile(f"{_LEFT}(?:{alts}){_RIGHT}")


def replace_terms(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    ordered = sorted(mapping, key=len, reverse=True)
    pattern = _term_pattern(ordered)

    def repl(m: re.Match[str]) -> str:
        return mapping[ordered[m.lastindex - 1]]

    return pattern.sub(repl, text)


# ---------- dollar scaling ----------

_NUM = r"\d{1,3}(?:,\d{3})+|\d+"
_DOLLAR = re.compile(
    rf"\$(?P<int>{_NUM})(?P<frac>\.\d+)?"
    # single-dollar range such as "$10-15 million" or "$2 to 3 billion": scale both ends
    rf"(?:(?P<sep>\s*(?:-|–|—|to)\s*)(?P<int2>{_NUM})(?P<frac2>\.\d+)?)?"
    r"(?P<unit>\s*(?:million|billion|thousand|trillion)\b)?",
    re.IGNORECASE,
)


def _sig3(value: Decimal) -> Decimal:
    if value == 0:
        return value
    exp = value.adjusted() - 2
    return value.quantize(Decimal(1).scaleb(exp), rounding=ROUND_HALF_UP)


def _format_amount(value: Decimal) -> str:
    if value == value.to_integral_value():
        return f"{int(value):,}"  # commas always, so a scaled figure never reads as a year
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def scale_dollars(text: str, factor: float) -> str:
    f = Decimal(str(factor))

    def scaled(int_part: str, frac: str | None) -> str:
        number = Decimal(int_part.replace(",", "") + (frac or ""))
        return _format_amount(_sig3(number * f))

    def repl(m: re.Match[str]) -> str:
        out = "$" + scaled(m.group("int"), m.group("frac"))
        if m.group("sep") is not None:
            out += m.group("sep") + scaled(m.group("int2"), m.group("frac2"))
        return out + (m.group("unit") or "")

    return _DOLLAR.sub(repl, text)


# ---------- year shift ----------

# numeric m/d/yy dates; four-digit years are handled by the standalone pattern
_NUMERIC_DATE_2Y = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{2})(?!\d)")
_FISCAL_2Y = re.compile(r"\b(FY\s?'?)(\d{2})(?!\d)")
_APOSTROPHE_YEAR = re.compile(r"(?<![\w'])'(\d{2})(?!\d)")
_YEAR_4 = re.compile(r"(?<![\d,.$])(19[89]\d|20[0-2]\d|2030)(?!\d|,\d|\.\d)")


def _shift_2y(yy: str, years: int) -> str:
    n = int(yy)
    full = 1900 + n if n >= 70 else 2000 + n
    return f"{(full + years) % 100:02d}"


def shift_years(text: str, years: int) -> str:
    def numeric(m: re.Match[str]) -> str:
        month, day = int(m.group(1)), int(m.group(2))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return m.group(0)
        return f"{m.group(1)}/{m.group(2)}/{_shift_2y(m.group(3), years)}"

    text = _NUMERIC_DATE_2Y.sub(numeric, text)
    text = _FISCAL_2Y.sub(lambda m: m.group(1) + _shift_2y(m.group(2), years), text)
    text = _APOSTROPHE_YEAR.sub(lambda m: "'" + _shift_2y(m.group(1), years), text)
    return _YEAR_4.sub(lambda m: str(int(m.group(1)) + years), text)


# ---------- levels ----------

def disguise_text(text: str, level: int, rules: DisguiseRules) -> str:
    if level <= 1:
        return text
    if level == 2:
        return replace_terms(text, rules.names)
    text = replace_terms(text, {**rules.names, **rules.structures})
    text = scale_dollars(text, rules.scale_factor)
    return shift_years(text, rules.year_shift)


def residual_identifiers(text: str, rules: DisguiseRules, level: int = 3) -> list[str]:
    """Listed identifiers still present, using the same matching rules as replacement."""
    if level < 2:
        return []
    structure_keys = set(rules.structures)
    found: list[str] = []
    for ident in rules.identifiers:
        if ident in structure_keys and level < 3:
            continue
        if _term_pattern([ident]).search(text):
            found.append(ident)
    return found
