from config import CONFIG
from generation.answer_generator import AnswerGenerator, build_prompt
from generation.llm_client import LLMOutput
from retrieval.hybrid_retriever import RetrievedChunk
from agent.agent import BaselineAgent
from agent.query_formulation import (
    QueryFormulator,
    build_query_formulation_prompt,
    parse_retrieval_query,
)
from evaluation.evaluator import Evaluator
from evaluation.answer_metrics import is_answer_correct


GOLD_MARKERS = (
    "gold_answer",
    "program_answer",
    "original_answer",
    "gold_evidence",
    "gold_context",
    "SECRET_GOLD",
)


class _ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.max_tokens_args = []

    def generate(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        self.max_tokens_args.append(max_tokens)
        text = self.replies.pop(0) if self.replies else "42.0"
        return LLMOutput(
            text=text,
            input_tokens=10,
            output_tokens=4,
            latency_seconds=0.01,
            raw={},
        )


class _Chunk:
    def __init__(self):
        self.chunk_id = "c0"
        self.page = 1
        self.pages = [1]
        self.document_id = "intel_2015"
        self.company = "Intel"
        self.year = 2015
        self.filing_type = "10-K"
        self.source_filename = "intel.pdf"
        self.text = "As of December 26, 2015, 15% of Intel total facilities square feet were leased."

    def matches_filters(self, filters):
        return True


class _Retr:
    def __init__(self):
        self.queries = []
        self.filters_seen = []
        self.chunks_by_id = {"c0": _Chunk()}
        self.retrieve_kwargs = []

    def retrieve(self, query, filters=None, dense_top_k=None, bm25_top_k=None, final_top_k=None):
        self.queries.append(query)
        self.filters_seen.append(filters)
        self.retrieve_kwargs.append(
            {
                "dense_top_k": dense_top_k,
                "bm25_top_k": bm25_top_k,
                "final_top_k": final_top_k,
            }
        )
        chunk = self.chunks_by_id["c0"]
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                rank=1,
                dense_score=0.9,
                bm25_score=0.8,
                hybrid_score=1.0,
                page=chunk.page,
                pages=list(chunk.pages),
                document_id=chunk.document_id,
                company=chunk.company,
                year=chunk.year,
                filing_type=chunk.filing_type,
                source_filename=chunk.source_filename,
                text=chunk.text,
            )
        ]


def _agent(replies):
    llm = _ScriptedLLM(replies)
    retriever = _Retr()
    generator = AnswerGenerator(cfg=CONFIG, llm=llm)
    agent = BaselineAgent(retriever, generator, CONFIG)
    return agent, llm, retriever


QUESTION = (
    "As of December 26, 2015, what percentage of Intel's total facilities, "
    "measured in square feet, were leased?"
)
FORMULATED = "Intel 2015 total facilities leased square feet"


def test_query_formulation_is_invoked_before_retrieval():
    agent, llm, retriever = _agent([FORMULATED, "15%"])
    record = agent.answer(QUESTION, "q-intel")
    assert llm.prompts[0] == build_query_formulation_prompt(QUESTION)
    assert QUESTION in llm.prompts[0]
    assert retriever.queries == [FORMULATED]
    assert record.retrieval_query == FORMULATED
    assert record.original_question == QUESTION
    assert record.number_of_llm_calls == 2


def test_query_formulation_receives_only_the_original_question():
    agent, llm, retriever = _agent([FORMULATED, "15%"])
    item = {
        "gold_answer": "SECRET_GOLD_ANSWER_XYZ",
        "program_answer": "SECRET_PROGRAM",
        "gold": {"evidence": "SECRET_GOLD_EVIDENCE_SHOULD_NEVER_APPEAR_IN_PROMPT"},
    }
    agent.answer(QUESTION, "q-intel")
    prompt = llm.prompts[0]
    assert QUESTION in prompt
    for marker in GOLD_MARKERS:
        assert marker.lower() not in prompt.lower()
    assert "SECRET_GOLD" not in prompt
    assert item["gold_answer"] not in prompt
    assert item["gold"]["evidence"] not in prompt
    assert "do not answer" in prompt.lower()


def test_formulated_query_is_passed_to_existing_retriever():
    agent, llm, retriever = _agent([FORMULATED, "15%"])
    agent.answer(QUESTION, "q-intel")
    assert retriever.queries[0] != QUESTION
    assert retriever.queries[0] == FORMULATED
    assert retriever.retrieve_kwargs[0] == {
        "dense_top_k": None,
        "bm25_top_k": None,
        "final_top_k": None,
    }


def test_answer_generation_still_uses_retrieved_evidence():
    agent, llm, retriever = _agent([FORMULATED, "15 percent"])
    record = agent.answer(QUESTION, "q-intel")
    gen_prompt = llm.prompts[1]
    assert gen_prompt == record.answer_generation_prompt
    assert QUESTION in gen_prompt
    assert "15% of Intel total facilities" in gen_prompt
    assert gen_prompt.strip().endswith("Answer:") or "\nAnswer:" in gen_prompt
    assert record.answer == "15 percent"
    assert record.retrieved_chunks[0]["chunk_id"] == "c0"
    assert record.retrieved_chunks[0]["rank"] == 1


def test_query_formulation_trace_is_logged_separately():
    agent, llm, retriever = _agent([FORMULATED, "15%"])
    record = agent.answer(QUESTION, "q-intel")
    payload = record.to_dict()
    assert payload["query_formulation_prompt"] == llm.prompts[0]
    assert payload["query_formulation_raw_output"] == FORMULATED
    assert payload["query_formulation_trace"]["retrieval_query"] == FORMULATED
    assert payload["query_formulation_latency_seconds"] >= 0
    assert payload["query_formulation_input_tokens"] == 10
    assert payload["query_formulation_output_tokens"] == 4
    assert payload["answer_generation_latency_seconds"] >= 0
    assert payload["answer_generation_input_tokens"] == 10
    assert payload["number_of_llm_calls"] == 2
    assert "gold_answer" not in payload["query_formulation_prompt"].lower()


def test_empty_formulation_falls_back_to_original_question():
    query = parse_retrieval_query("   \n```\n```", QUESTION)
    assert query == QUESTION
    formulator = QueryFormulator(llm=_ScriptedLLM([""]), cfg=CONFIG)
    result = formulator.formulate(QUESTION)
    assert result.retrieval_query == QUESTION
    assert result.llm_calls == 1


def test_metadata_filters_still_come_from_the_original_question():
    agent, llm, retriever = _agent(["facilities leased square feet", "15%"])
    record = agent.answer(QUESTION, "q-intel")
    assert record.metadata_filters.get("year") == 2015
    assert retriever.queries == ["facilities leased square feet"]


def test_evaluation_still_uses_gold_only_after_generation():
    agent, llm, retriever = _agent([FORMULATED, "15"])
    record = agent.answer(QUESTION, "q-intel")
    payload = record.to_dict()
    assert "gold_answer" not in payload
    from experiments.run_t2_ragbench import attach_eval_metadata

    attached = attach_eval_metadata(
        payload,
        {
            "dataset": "FinQA",
            "gold_answer": "15",
            "gold_evidence_reference": {"pages": [1]},
        },
    )
    assert attached["gold_answer"] == "15"
    assert attached["dataset"] == "FinQA"
    assert is_answer_correct(record.answer, "15")
    assert Evaluator(ks=[1, 3, 5, 10]).ks == [1, 3, 5, 10]


def test_whitespace_helper_unchanged_for_self_rag():
    """Whitespace helper remains non-LLM; Self-RAG retrieve uses formulate_retrieval_query."""
    agent, llm, retriever = _agent([FORMULATED, "15%"])
    assert agent.formulate_query("  What is EPS?  ") == "What is EPS?"
    assert llm.prompts == []
    assert retriever.queries == []
