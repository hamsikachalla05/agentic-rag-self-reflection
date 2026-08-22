from self_rag.labels import (
    EVIDENCE_SUPPORT_LABELS,
    RETRIEVAL_DECISION_LABELS,
    RETRIEVAL_RELEVANCE_LABELS,
    USEFULNESS_LABELS,
    parse_followup_query,
    parse_label,
)


def test_parses_valid_labels():
    assert parse_label("NEED_RETRIEVAL", RETRIEVAL_DECISION_LABELS, "NEED_RETRIEVAL") == (
        "NEED_RETRIEVAL",
        True,
    )
    assert parse_label("Label: no_retrieval\n", RETRIEVAL_DECISION_LABELS, "NEED_RETRIEVAL") == (
        "NO_RETRIEVAL",
        True,
    )
    assert parse_label("IRRELEVANT", RETRIEVAL_RELEVANCE_LABELS, "RELEVANT") == ("IRRELEVANT", True)
    assert parse_label("supported.", EVIDENCE_SUPPORT_LABELS, "SUPPORTED") == ("SUPPORTED", True)
    assert parse_label("NOT_USEFUL", USEFULNESS_LABELS, "USEFUL") == ("NOT_USEFUL", True)
    assert parse_label("The draft is USEFUL", USEFULNESS_LABELS, "USEFUL") == ("USEFUL", True)


def test_not_useful_is_not_confused_with_useful():
    label, ok = parse_label("NOT USEFUL", USEFULNESS_LABELS, "USEFUL")
    assert ok is True
    assert label == "NOT_USEFUL"


def test_malformed_output_fails_safely():
    assert parse_label("", RETRIEVAL_DECISION_LABELS, "NEED_RETRIEVAL") == ("NEED_RETRIEVAL", False)
    assert parse_label("   ", RETRIEVAL_RELEVANCE_LABELS, "RELEVANT") == ("RELEVANT", False)
    assert parse_label("I am not sure", EVIDENCE_SUPPORT_LABELS, "SUPPORTED") == ("SUPPORTED", False)
    assert parse_label("maybe?", USEFULNESS_LABELS, "USEFUL") == ("USEFUL", False)
    assert parse_label(None, RETRIEVAL_DECISION_LABELS, "NEED_RETRIEVAL")[1] is False  # type: ignore[arg-type]


def test_followup_query_optional():
    assert parse_followup_query("IRRELEVANT\nQUERY: Apple net sales 2001") == "Apple net sales 2001"
    assert parse_followup_query("RELEVANT") is None
    assert parse_followup_query("QUERY: RELEVANT") is None
