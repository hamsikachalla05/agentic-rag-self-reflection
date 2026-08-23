from agent.agent import BaselineAgent


class _Dummy:
    chunks_by_id = {}


def _agent(companies=None, doc_ids=None):
    agent = BaselineAgent.__new__(BaselineAgent)
    agent.retriever = _Dummy()
    agent.known_companies = companies or ["Northstar Financial Inc", "CedarTech PLC"]
    agent.known_document_ids = doc_ids or []
    return agent


def test_extracts_year_and_company_when_explicit():
    agent = _agent()
    filters = agent.extract_metadata_filters(
        "What were Northstar Financial Inc. total revenues in 2023?"
    )
    assert filters["year"] == 2023
    assert filters["company"] == "Northstar Financial Inc"


def test_does_not_invent_company_or_year():
    agent = _agent()
    filters = agent.extract_metadata_filters("What was operating income?")
    assert "company" not in filters
    assert "year" not in filters


def test_query_is_the_question():
    """Whitespace helper used by frozen Self-RAG; LLM formulation is separate."""
    agent = _agent()
    assert agent.formulate_query("  What is EPS?  ") == "What is EPS?"
