import inspect

from config import CONFIG
from generation.answer_generator import build_prompt
from retrieval.hybrid_retriever import RetrievedChunk
from self_rag.controller import SelfRAGAgent
from self_rag.prompts import (
    answer_usefulness_prompt,
    evidence_support_prompt,
    retrieval_decision_prompt,
    retrieval_relevance_prompt,
    revision_prompt,
)


FORBIDDEN = (
    "gold_answer",
    "program_answer",
    "original_answer",
    "gold_evidence",
    "gold_context",
    "SECRET_GOLD",
)


def _chunk():
    return RetrievedChunk(
        chunk_id="c0",
        rank=1,
        dense_score=1.0,
        bm25_score=1.0,
        hybrid_score=1.0,
        page=1,
        pages=[1],
        document_id="doc",
        company="Acme",
        year=2015,
        filing_type="10-K",
        source_filename="page.pdf",
        text="Revenue was 10.0.",
    )


def test_reflection_prompts_do_not_include_gold_fields():
    question = "What was revenue?"
    chunk = _chunk()
    prompts = [
        retrieval_decision_prompt(question),
        retrieval_relevance_prompt(question, [chunk], CONFIG),
        evidence_support_prompt(question, [chunk], CONFIG),
        answer_usefulness_prompt(question, [chunk], "10.0", CONFIG),
        revision_prompt(question, [chunk], "10.0", CONFIG),
        build_prompt(question, [chunk], CONFIG),
    ]
    for prompt in prompts:
        lowered = prompt.lower()
        for marker in FORBIDDEN:
            assert marker.lower() not in lowered
        assert "SECRET_GOLD" not in prompt


def test_self_rag_runner_does_not_pass_gold_into_agent():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "experiments" / "run_t2_self_rag.py").read_text(
        encoding="utf-8"
    )
    assert "agent.answer(question, qid)" in source
    assert "agent.answer(question, qid, " not in source
    assert "attach_eval_metadata" in source
