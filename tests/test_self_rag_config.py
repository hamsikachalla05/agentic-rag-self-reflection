from config import CONFIG
from generation.answer_generator import AnswerGenerator
from generation.llm_client import LLMOutput
from retrieval.hybrid_retriever import RetrievedChunk
from self_rag.config import SelfRAGConfig
from self_rag.controller import SelfRAGAgent


class _Chunk:
    chunk_id = "c0"
    page = 1
    pages = [1]
    document_id = "doc"
    company = "Acme"
    year = 2015
    filing_type = "10-K"
    source_filename = "page.pdf"
    text = "Revenue was 10.0 in 2015."

    def matches_filters(self, filters):
        return True


class _LLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        if "NEED_RETRIEVAL" in prompt and "Label:" in prompt and "Evidence:" not in prompt:
            return LLMOutput("NEED_RETRIEVAL", 1, 1, 0.0, {})
        if "RELEVANT if any retrieved chunk" in prompt:
            return LLMOutput("RELEVANT", 1, 1, 0.0, {})
        if "SUPPORTED if the evidence contains" in prompt:
            return LLMOutput("SUPPORTED", 1, 1, 0.0, {})
        if "USEFUL if the draft answers" in prompt:
            return LLMOutput("USEFUL", 1, 1, 0.0, {})
        return LLMOutput("10.0", 1, 1, 0.0, {})


class _Retr:
    chunks_by_id = {"c0": _Chunk()}

    def retrieve(self, query, filters=None, dense_top_k=None, bm25_top_k=None, final_top_k=None):
        c = _Chunk()
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                rank=1,
                dense_score=1.0,
                bm25_score=1.0,
                hybrid_score=1.0,
                page=c.page,
                pages=list(c.pages),
                document_id=c.document_id,
                company=c.company,
                year=c.year,
                filing_type=c.filing_type,
                source_filename=c.source_filename,
                text=c.text,
            )
        ]


def _run(cfg: SelfRAGConfig):
    llm = _LLM()
    agent = SelfRAGAgent(_Retr(), AnswerGenerator(cfg=CONFIG, llm=llm), CONFIG, cfg)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    return record, llm


def test_each_component_can_be_disabled_independently():
    full, llm_full = _run(SelfRAGConfig())
    assert any(s["component"] == "retrieval_decision" and not s["skipped"] for s in full.reflection_trace)
    assert any(s["component"] == "retrieval_relevance" and not s["skipped"] for s in full.reflection_trace)
    assert any(s["component"] == "evidence_support" and not s["skipped"] for s in full.reflection_trace)
    assert any(s["component"] == "answer_usefulness" and not s["skipped"] for s in full.reflection_trace)

    no_dec, llm_dec = _run(SelfRAGConfig(retrieval_decision=False))
    assert all(
        s["skipped"] for s in no_dec.reflection_trace if s["component"] == "retrieval_decision"
    )
    assert any("NEED_RETRIEVAL if answering" in p for p in llm_full.prompts)
    assert not any("NEED_RETRIEVAL if answering" in p for p in llm_dec.prompts)

    no_rel, llm_rel = _run(SelfRAGConfig(retrieval_relevance=False))
    assert all(
        s["skipped"] for s in no_rel.reflection_trace if s["component"] == "retrieval_relevance"
    )
    assert not any("RELEVANT if any retrieved chunk" in p for p in llm_rel.prompts)

    no_sup, llm_sup = _run(SelfRAGConfig(evidence_support=False))
    assert all(
        s["skipped"] for s in no_sup.reflection_trace if s["component"] == "evidence_support"
    )
    assert not any("SUPPORTED if the evidence contains" in p for p in llm_sup.prompts)

    no_use, llm_use = _run(SelfRAGConfig(answer_usefulness=False))
    assert all(
        s["skipped"] for s in no_use.reflection_trace if s["component"] == "answer_usefulness"
    )
    assert not any("USEFUL if the draft answers" in p for p in llm_use.prompts)
    assert no_use.answer_usefulness_decision == "USEFUL"


def test_full_self_rag_enables_all_four_flags():
    cfg = SelfRAGConfig()
    assert cfg.retrieval_decision is True
    assert cfg.retrieval_relevance is True
    assert cfg.evidence_support is True
    assert cfg.answer_usefulness is True
