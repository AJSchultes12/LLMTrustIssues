from llmtrust.leakscan import scan_reasoning

TERMS = ["bankruptcy", "Raptor", "restatement"]


def test_scan_flags_only_terms_absent_from_prompt():
    prompt = "The memo discusses the Raptor vehicles and accounting."
    reasoning = "Raptor hedges look fragile; this could end in bankruptcy."
    assert scan_reasoning(reasoning, prompt, TERMS) == ["bankruptcy"]


def test_scan_is_case_insensitive():
    assert scan_reasoning("BANKRUPTCY looms", "nothing here", TERMS) == ["bankruptcy"]
    assert scan_reasoning("a Restatement", "nothing here", TERMS) == ["restatement"]


def test_scan_is_whole_word():
    assert scan_reasoning("a nonbankruptcyish outcome", "nothing", TERMS) == []
    assert scan_reasoning("bankruptcy's shadow", "nothing", TERMS) == ["bankruptcy"]


def test_scan_prompt_match_is_case_insensitive_and_whole_word():
    assert scan_reasoning("bankruptcy", "BANKRUPTCY is mentioned", TERMS) == []
    assert scan_reasoning("bankruptcy", "nonbankruptcyish", TERMS) == ["bankruptcy"]


def test_scan_returns_each_term_once_in_term_order():
    reasoning = "restatement, bankruptcy, bankruptcy again, restatement"
    assert scan_reasoning(reasoning, "", TERMS) == ["bankruptcy", "restatement"]
    assert scan_reasoning(reasoning, "", ["bankruptcy", "bankruptcy"]) == ["bankruptcy"]
