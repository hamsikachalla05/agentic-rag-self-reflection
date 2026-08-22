from generation.answer_generator import build_prompt
from retrieval.hybrid_retriever import RetrievedChunk
from config import CONFIG
from agent.agent import BaselineAgent
from pathlib import Path


def test_t2_gold_fields_are_not_in_generation_prompt():
    chunk = RetrievedChunk(
        chunk_id="ctx::p1::c0",
        rank=1,
        dense_score=1.0,
        bm25_score=1.0,
        hybrid_score=1.0,
        page=1,
        pages=[1],
        document_id="finqa_test_ctx_1",
        company="Analog Devices",
        year=2009,
        filing_type="annual report",
        source_filename="finqa_test_ctx_1.pdf",
        text="Interest expense would change by $ 3.8 million.",
    )
    prompt = build_prompt(
        "What was Analog Devices reported interest expense for fiscal year 2009?",
        [chunk],
        CONFIG,
    )
    assert "gold_answer" not in prompt
    assert "program_answer" not in prompt
    assert "original_answer" not in prompt
    assert "gold_evidence" not in prompt
    assert "gold_context" not in prompt
    # Retrieved PDF text may be present; the withheld gold string is not injected.
    assert "SECRET_GOLD_CONTEXT_SHOULD_NOT_APPEAR" not in prompt


def test_agent_answer_only_takes_question_and_id():
    import inspect

    params = list(inspect.signature(BaselineAgent.answer).parameters)
    assert params == ["self", "question", "question_id"]


def test_t2_runner_does_not_pass_gold_into_agent():
    source = (Path(__file__).resolve().parents[1] / "experiments" / "run_t2_ragbench.py").read_text(
        encoding="utf-8"
    )
    assert "agent.answer(question, qid)" in source
    assert "agent.answer(question, qid, " not in source
    assert "attach_eval_metadata" in source
    assert 'out["gold_answer"]' in source
