from dataclasses import replace

from config import CONFIG
from generation.answer_generator import AnswerGenerator
from generation.llm_client import LLMOutput
from retrieval.hybrid_retriever import RetrievedChunk
from self_rag.config import SelfRAGConfig
from self_rag.controller import SelfRAGAgent


class _Chunk:
    def __init__(self, chunk_id="c0", company="Acme", year=2015, document_id="doc"):
        self.chunk_id = chunk_id
        self.page = 1
        self.pages = [1]
        self.document_id = document_id
        self.company = company
        self.year = year
        self.filing_type = "10-K"
        self.source_filename = "page.pdf"
        self.text = "Revenue was 10.0 in 2015."

    def matches_filters(self, filters):
        return True


class _ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "USEFUL"
        return LLMOutput(text=text, input_tokens=2, output_tokens=1, latency_seconds=0.0, raw={})


class _Retr:
    def __init__(self, n_calls_holder):
        self.n_calls_holder = n_calls_holder
        self.chunks_by_id = {"c0": _Chunk()}

    def retrieve(self, query, filters=None, dense_top_k=None, bm25_top_k=None, final_top_k=None):
        self.n_calls_holder["n"] += 1
        chunk = self.chunks_by_id["c0"]
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                rank=1,
                dense_score=1.0,
                bm25_score=1.0,
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


def _agent(replies, rag_cfg, n_calls_holder=None):
    n_calls_holder = n_calls_holder or {"n": 0}
    llm = _ScriptedLLM(replies)
    generator = AnswerGenerator(cfg=CONFIG, llm=llm)
    retriever = _Retr(n_calls_holder)
    agent = SelfRAGAgent(retriever, generator, CONFIG, rag_cfg)
    return agent, llm, n_calls_holder


def test_max_retrieval_iterations_are_enforced():
    rag = SelfRAGConfig(
        retrieval_decision=True,
        retrieval_relevance=True,
        evidence_support=True,
        answer_usefulness=True,
        max_retrieval_iterations=2,
        max_answer_attempts=1,
        max_llm_calls=20,
    )
    # Always IRRELEVANT / UNSUPPORTED would otherwise loop.
    replies = [
        "NEED_RETRIEVAL",
        "IRRELEVANT\nQUERY: more table values",
        "UNSUPPORTED",
        "IRRELEVANT",
        "UNSUPPORTED",
        "10.0",  # generation
        "USEFUL",
    ]
    holder = {"n": 0}
    agent, llm, holder = _agent(replies, rag, holder)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert holder["n"] <= rag.max_retrieval_iterations
    assert record.number_of_retrieval_iterations <= rag.max_retrieval_iterations
    assert record.number_of_llm_calls <= rag.max_llm_calls


def test_not_useful_with_no_chunks_triggers_retrieval():
    rag = SelfRAGConfig(
        max_retrieval_iterations=2,
        max_answer_attempts=2,
        max_llm_calls=12,
    )
    replies = [
        "NO_RETRIEVAL",
        "The retrieved evidence is insufficient to answer this question",
        "NOT_USEFUL",
        "10.0",
        "USEFUL",
    ]
    holder = {"n": 0}
    agent, llm, holder = _agent(replies, rag, holder)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert holder["n"] >= 1
    assert record.number_of_retrieval_iterations >= 1
    assert record.retrieved_chunks
    assert record.answer == "10.0"


def test_max_llm_calls_cap_skips_further_reflection():
    rag = SelfRAGConfig(
        max_retrieval_iterations=2,
        max_answer_attempts=2,
        max_llm_calls=2,
    )
    replies = ["NEED_RETRIEVAL", "10.0"]  # reserve the second call for generation
    holder = {"n": 0}
    agent, llm, holder = _agent(replies, rag, holder)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert record.number_of_llm_calls <= rag.max_llm_calls
    assert len(llm.prompts) <= rag.max_llm_calls
