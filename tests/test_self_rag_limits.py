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
        self.text = "Revenue was 10.0 in 2015. " + ("body " * 400)

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
        self.queries = []
        self.chunks_by_id = {"c0": _Chunk()}

    def retrieve(self, query, filters=None, dense_top_k=None, bm25_top_k=None, final_top_k=None):
        self.n_calls_holder["n"] += 1
        self.queries.append(query)
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
    return agent, llm, n_calls_holder, retriever


def test_max_hybrid_retrieves_is_one_when_unsupported():
    rag = SelfRAGConfig(
        max_retrieval_iterations=1,
        max_evidence_refinements=2,
        max_answer_attempts=1,
        max_llm_calls=20,
    )
    replies = [
        "Acme revenue 2015",
        "NEED_RETRIEVAL",
        "IRRELEVANT",
        "UNSUPPORTED",
        "UNSUPPORTED",
        "UNSUPPORTED",
        "10.0",
        "USEFUL",
    ]
    holder = {"n": 0}
    agent, llm, holder, retriever = _agent(replies, rag, holder)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert holder["n"] == 1
    assert retriever.queries == ["Acme revenue 2015"]
    assert record.number_of_retrieval_iterations == 1
    assert record.number_of_evidence_refinements <= rag.max_evidence_refinements
    assert record.retrieved_chunks[0]["rank"] == 1


def test_no_retrieval_decision_does_not_drop_rank1():
    rag = SelfRAGConfig(max_llm_calls=12, max_answer_attempts=1)
    replies = [
        "Acme revenue 2015",
        "NO_RETRIEVAL",
        "RELEVANT",
        "SUPPORTED",
        "10.0",
        "USEFUL",
    ]
    holder = {"n": 0}
    agent, llm, holder, retriever = _agent(replies, rag, holder)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert holder["n"] >= 1
    assert record.retrieved_chunks
    assert record.retrieved_chunks[0]["rank"] == 1
    assert record.answer == "10.0"


def test_max_llm_calls_cap_skips_further_reflection():
    rag = SelfRAGConfig(
        max_retrieval_iterations=1,
        max_answer_attempts=2,
        max_llm_calls=2,
    )
    replies = ["Acme revenue 2015", "NEED_RETRIEVAL", "10.0"]
    holder = {"n": 0}
    agent, llm, holder, retriever = _agent(replies, rag, holder)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    stage_calls = record.number_of_llm_calls - 1  # exclude query formulation
    assert stage_calls <= rag.max_llm_calls
    assert record.number_of_llm_calls <= rag.max_llm_calls + 1
