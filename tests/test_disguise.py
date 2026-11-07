from datetime import date
from pathlib import Path

import pytest

from llmtrust.disguise import (
    DisguiseRules,
    audit_text,
    disguise_text,
    load_rules,
    replace_terms,
    residual_identifiers,
    scale_dollars,
    shift_years,
)

ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = ROOT / "cases" / "enron.disguise.yaml"


@pytest.fixture
def rules() -> DisguiseRules:
    return DisguiseRules(
        names={
            "Enron Corp.": "the Company",
            "Enron Corp/OR/": "the Company",
            "Enron": "the Company",
            "Lay": "the chairman",
            "Houston": "a Gulf Coast city",
        },
        structures={"LJM1": "Partnership One", "LJM": "Partnership", "Raptors": "Vehicles", "Raptor": "Vehicle"},
        identifiers=["Enron Corp.", "Enron Corp/OR/", "Enron", "Lay", "Houston",
                     "LJM1", "LJM", "Raptors", "Raptor"],
        scale_factor=0.37,
        year_shift=28,
    )


@pytest.fixture
def real_rules() -> DisguiseRules:
    return load_rules(RULES_PATH)


def test_level1_unchanged(rules):
    text = "Enron Corp. paid $100 million in 2001 to LJM1."
    assert disguise_text(text, 1, rules) == text


def test_names_replaced_longest_first(rules):
    out = disguise_text("Enron Corp. is Enron, and Enron Corp. said so.", 2, rules)
    assert "Enron" not in out
    assert out == "the Company is the Company, and the Company said so."


def test_word_boundaries_and_possessives(rules):
    assert disguise_text("relay the memo to Lay", 2, rules) == "relay the memo to the chairman"
    assert disguise_text("Layton and Lay's office", 2, rules) == "Layton and the chairman's office"
    assert disguise_text("Enron's board", 2, rules) == "the Company's board"
    out = disguise_text("ENRON CORP/OR/ filed a 10-K", 2, rules)
    assert "ENRON" not in out.upper().replace("THE COMPANY", "")
    assert out == "the Company filed a 10-K"
    assert disguise_text("they lay off staff", 2, rules) == "they lay off staff"


def test_long_keys_case_insensitive_short_keys_case_sensitive(rules):
    assert disguise_text("HOUSTON and houston", 2, rules) == "a Gulf Coast city and a Gulf Coast city"
    assert disguise_text("LAY and lay", 2, rules) == "LAY and lay"
    out = disguise_text("ljm and LJM and LJM1 and Raptors and Raptor", 3, rules)
    assert out == "ljm and Partnership and Partnership One and Vehicles and Vehicle"


def test_level2_keeps_numbers_and_dates(rules):
    text = "On August 14, 2001 Enron paid $100 million (12%) for 5,000 shares at 6/28/99."
    out = disguise_text(text, 2, rules)
    assert out == "On August 14, 2001 the Company paid $100 million (12%) for 5,000 shares at 6/28/99."
    assert "LJM1" in disguise_text("LJM1 and Enron", 2, rules)  # structures only at level 3


def test_level3_applies_structures(rules):
    assert disguise_text("Enron sold to LJM1", 3, rules) == "the Company sold to Partnership One"


def test_level3_scales_dollars_by_one_factor(rules):
    out = disguise_text("Paid $100 million, then $50 million, with a $1.2 billion note, 12% or 3,000,000 shares.", 3, rules)
    assert "$37 million" in out
    assert "$18.5 million" in out
    assert "$0.444 billion" in out
    assert "12%" in out
    assert "3,000,000 shares" in out


def test_scale_dollars_forms():
    assert scale_dollars("$100 million", 0.37) == "$37 million"
    assert scale_dollars("$1,234,567", 0.37) == "$457,000"
    assert scale_dollars("$5,400 million", 0.37) == "$2,000 million"
    assert scale_dollars("$2.5 Billion", 0.37) == "$0.925 Billion"
    assert scale_dollars("cost $10, then $20.", 0.5) == "cost $5, then $10."
    assert scale_dollars("no money here: 100 shares, 25%", 0.37) == "no money here: 100 shares, 25%"


def test_scale_dollars_single_dollar_ranges(rules):
    assert scale_dollars("$10-15 million", 0.37) == "$3.7-5.55 million"
    assert scale_dollars("$200–300 million", 0.37) == "$74–111 million"
    assert scale_dollars("$2 to 3 billion", 0.37) == "$0.74 to 1.11 billion"
    assert scale_dollars("$10 - 15 million", 0.37) == "$3.7 - 5.55 million"
    # both ends use the same factor: ratio preserved within rounding
    out = scale_dollars("$200-300 million", 0.37)
    lo, hi = (float(x) for x in out.lstrip("$").split(" ")[0].split("-"))
    assert abs(hi / lo - 300 / 200) < 0.01
    # no original figure survives, and the level 3 pipeline uses the same path
    assert "15" not in disguise_text("a $10-15 million fee", 3, rules).replace("5.55", "")
    assert disguise_text("a $10-15 million fee", 3, rules) == "a $3.7-5.55 million fee"
    # separate dollar signs on each end still scale independently as before
    assert scale_dollars("$10-$15 million", 0.37) == "$3.7-$5.55 million"


def test_level3_year_shift_preserves_gaps_and_weekdays(rules):
    out = disguise_text("August 14, 2001 and August 22, 2001", 3, rules)
    assert out == "August 14, 2029 and August 22, 2029"
    a = date(2029, 8, 14)
    b = date(2029, 8, 22)
    assert (b - a).days == 8
    assert a.weekday() == date(2001, 8, 14).weekday()
    assert b.weekday() == date(2001, 8, 22).weekday()


def test_year_shift_date_forms():
    assert shift_years("2001-08-14", 28) == "2029-08-14"
    assert shift_years("Aug. 14, 2001", 28) == "Aug. 14, 2029"
    assert shift_years("14 Aug 2001", 28) == "14 Aug 2029"
    assert shift_years("6/28/1999", 28) == "6/28/2027"
    assert shift_years("1/5/05", 28) == "1/5/33"
    assert shift_years("12/31/70", 28) == "12/31/98"
    assert shift_years("the 1979 report", 28) == "the 1979 report"  # outside 1980-2030


def test_short_dates_and_fiscal_labels(rules):
    text = "Memo of 6/28/99 on FY2001, FY01, Q2 2001 and the '01 plan; also 2000 and 1999."
    out = disguise_text(text, 3, rules)
    for old in ("1999", "2000", "2001"):
        assert old not in out
    assert "6/28/27" in out
    assert "FY2029" in out
    assert "FY29" in out
    assert "Q2 2029" in out
    assert "'29 plan" in out
    assert "2028 and 2027" in out


def test_year_shift_leaves_numbers_that_are_not_years():
    text = "Paid 1,999 units, ratio 2001.5, CIK 0001024401, accession 0001-00-1, $2,001."
    assert shift_years(text, 28) == text


def test_dollar_scaling_does_not_create_year_lookalikes(rules):
    # $5,400 million * 0.37 is about 2,000: must keep a comma so it is not read as a year
    out = disguise_text("It paid $5,400 million in 2001.", 3, rules)
    assert out == "It paid $2,000 million in 2029."


def test_replace_terms_standalone():
    assert replace_terms("Enron Corp. and Enron", {"Enron": "X", "Enron Corp.": "Y"}) == "Y and X"
    assert replace_terms("anything", {}) == "anything"


def test_residual_identifiers_empty_after_level2(rules):
    text = "Enron Corp. and Lay met in Houston; Enron's board and ENRON CORP/OR/ agreed."
    assert residual_identifiers(text, rules, 2) == ["Enron Corp.", "Enron Corp/OR/", "Enron", "Lay", "Houston"]
    assert residual_identifiers(disguise_text(text, 2, rules), rules, 2) == []
    assert residual_identifiers(text, rules, 1) == []


def test_residual_identifiers_reports_leaks(rules):
    assert residual_identifiers("Lay and enron", rules, 2) == ["Enron", "Lay"] or \
        sorted(residual_identifiers("Lay and enron", rules, 2)) == ["Enron", "Lay"]
    assert residual_identifiers("relay Layton", rules, 2) == []
    # structure identifiers only count at level 3
    assert residual_identifiers("LJM1 and Raptors", rules, 2) == []
    assert sorted(residual_identifiers("LJM1 and Raptors", rules, 3)) == ["LJM1", "Raptors"]
    assert residual_identifiers(disguise_text("LJM1 and Raptors and Raptor", 3, rules), rules, 3) == []


def test_real_rules_file_loads(real_rules):
    r = real_rules
    assert r.scale_factor == 0.37
    assert r.year_shift == 28
    required_names = ["Enron Corp.", "Enron", "EnronOnline", "Kenneth Lay", "Lay", "Jeffrey Skilling",
                      "Skilling", "Andrew Fastow", "Fastow", "Sherron Watkins", "Watkins",
                      "Arthur Andersen", "Andersen", "Vinson & Elkins", "Houston", "Portland General"]
    for k in required_names:
        assert k in r.names
    required_structures = ["LJM1", "LJM2", "LJM Cayman", "LJM", "Raptors", "Raptor", "Chewco", "JEDI",
                           "Condor", "Whitewing", "Marlin", "Osprey", "Braveheart", "Rawhide", "Talon",
                           "Porcupine", "Yosemite"]
    for k in required_structures:
        assert k in r.structures
    for k in list(r.names) + list(r.structures):
        assert k in r.identifiers


def test_audit_text_substring_for_long_keys(rules):
    # the replacer needs word boundaries; the audit does not
    assert audit_text("EnronBroadband and HOUSTONIAN", rules, 2) == ["Enron", "Houston"]
    assert audit_text("the Company in a Gulf Coast city", rules, 2) == []
    assert audit_text("", rules, 3) == []
    assert audit_text("Enron", rules, 1) == []


def test_audit_text_short_keys_only_inside_email_addresses(rules):
    assert audit_text("they lay off staff; LAY said", rules, 3) == []
    assert audit_text("write to k.lay@corp.example", rules, 2) == ["Lay"]
    assert audit_text("write to LAY@corp.example", rules, 2) == ["Lay"]
    assert audit_text("write to relay@corp.example", rules, 2) == []


def test_audit_text_structures_only_at_level3(rules):
    assert audit_text("LJM1 and the Raptors", rules, 2) == []
    assert audit_text("the RAPTORS", rules, 3) == ["Raptors", "Raptor"]
    assert audit_text("mail ljm@x.example", rules, 3) == ["LJM"]


AUDIT_CASES = [
    "From: kenneth.lay@enron.com\nTo: jeff.skilling@enron.com, steven.kean@enron.com\n"
    "Cc: klay@enron.com; jskilling@ENRON.COM; andrew.fastow@enron.com; afastow@enron.com; "
    "sherron.watkins@enron.com; swatkins@enron.com\nSubject: Q2",
    "Kenneth L. Lay\tChairman\nJeffrey K. Skilling\tPresident\nK. Lay\tDirector",
    "Visit EnronCredit.com or Enron Credit, EnronOnline and EOL; ticker ENE.",
    "Signed K. LAY. Listed as LAY, KENNETH and Skilling, Jeffrey and Fastow, Andrew.",
    "Kean, Whalley, Causey, Koenig, Kopper, Glisan and Baxter (Steven Kean, Greg Whalley, "
    "Richard Causey, Mark Koenig, Michael Kopper, Ben Glisan, Cliff Baxter) discussed "
    "Azurix and Dabhol with Skadden.",
]


@pytest.mark.parametrize("text", AUDIT_CASES)
@pytest.mark.parametrize("level", [2, 3])
def test_real_rules_blind_spots_come_out_clean(real_rules, text, level):
    assert audit_text(text, real_rules, level) != []  # the raw text does leak
    out = disguise_text(text, level, real_rules)
    assert audit_text(out, real_rules, level) == [], out
    assert residual_identifiers(out, real_rules, level) == [], out


def test_real_rules_html_table_cells_come_out_clean(real_rules):
    from llmtrust.sources import _strip_html
    html = ("<table><tr><th>Name</th><th>Title</th></tr>"
            "<tr><td>Kenneth L. Lay</td><td>Chairman</td></tr>"
            "<tr><td>LAY, KENNETH</td><td>Director</td></tr></table>")
    text = _strip_html(html)
    for level in (2, 3):
        out = disguise_text(text, level, real_rules)
        assert audit_text(out, real_rules, level) == [], out


def test_real_rules_clean_after_levels(real_rules):
    text = ("Enron Corp. (ENRON CORP/OR/), Houston: Kenneth Lay, Jeffrey Skilling and Andrew Fastow; "
            "Sherron Watkins wrote to Lay about Arthur Andersen and Vinson & Elkins. "
            "EnronOnline, Portland General, LJM1, LJM2, LJM Cayman, Raptors, Raptor, Chewco, JEDI, "
            "Condor, Whitewing, Marlin, Osprey, Braveheart, Rawhide, Talon, Porcupine, Yosemite.")
    assert residual_identifiers(text, real_rules, 2) != []
    out2 = disguise_text(text, 2, real_rules)
    assert residual_identifiers(out2, real_rules, 2) == []
    assert "LJM1" in out2
    out3 = disguise_text(text, 3, real_rules)
    assert residual_identifiers(out3, real_rules, 3) == []
