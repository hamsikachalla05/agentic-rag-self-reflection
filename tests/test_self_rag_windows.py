from config import CONFIG
from generation.answer_generator import AnswerGenerator, build_prompt
from generation.llm_client import LLMOutput
from retrieval.hybrid_retriever import RetrievedChunk
from self_rag.config import SelfRAGConfig
from self_rag.controller import SelfRAGAgent
from self_rag.windows import EvidenceWindowTracker


def _rc(chunk_id="c0", rank=1, document_id="doc", text="", **kwargs):
    return RetrievedChunk(
        chunk_id=chunk_id,
        rank=rank,
        dense_score=1.0,
        bm25_score=1.0,
        hybrid_score=1.0,
        page=1,
        pages=[1],
        document_id=document_id,
        company="Acme",
        year=2015,
        filing_type="10-K",
        source_filename="page.pdf",
        text=text,
    )


class _IndexChunk:
    def __init__(self, chunk_id, document_id, text):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.page = 1
        self.pages = [1]
        self.company = "Acme"
        self.year = 2015
        self.filing_type = "10-K"
        self.source_filename = "page.pdf"
        self.text = text

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
    def __init__(self, hits, index_chunks=None):
        self.hits = hits
        self.queries = []
        self.chunks_by_id = {c.chunk_id: c for c in (index_chunks or hits)}

    def retrieve(self, query, filters=None, dense_top_k=None, bm25_top_k=None, final_top_k=None):
        self.queries.append(query)
        return list(self.hits)


def test_window_slide_changes_text_and_keeps_original_chunk_id():
    window = 700
    text = ("AAA" * 300) + ("BBB" * 300) + ("CCC" * 300)
    chunk = _rc(text=text)
    tracker = EvidenceWindowTracker([chunk], window_chars=window)
    first = tracker.current_view()[0].text
    assert first.startswith("AAA")
    assert "BBB" not in first[:50]
    assert tracker.advance() is True
    second = tracker.current_view()[0].text
    assert second != first
    assert tracker.history[-1]["windows"][0]["char_start"] == 700
    assert "BBB" in second
    assert tracker.original[0].chunk_id == "c0"
    assert tracker.original[0].rank == 1


def test_refinement_uses_same_document_sibling_not_new_search():
    first = _rc(chunk_id="doc::p1::c0::aaaa", text="START" + ("x" * 200))
    sibling = _IndexChunk("doc::p1::c1::bbbb", "doc", "LATER_TABLE 999.0 " + ("y" * 100))
    tracker = EvidenceWindowTracker(
        [first],
        window_chars=700,
        chunks_by_id={first.chunk_id: first, sibling.chunk_id: sibling},
    )
    moved = tracker.advance()
    assert moved is True
    view = tracker.current_view()[0]
    assert "LATER_TABLE" in view.text
    assert tracker.original[0].chunk_id == "doc::p1::c0::aaaa"


def test_controller_refinement_changes_generation_evidence_without_requery():
    window = CONFIG.max_chars_per_chunk_in_prompt
    text = ("FIRSTWIN" * 90) + ("SECONDWIN" * 90)
    hit = _rc(text=text)
    retriever = _Retr([hit], [hit])
    replies = [
        "NEED_RETRIEVAL",
        "RELEVANT",
        "UNSUPPORTED",
        "SUPPORTED",
        "the later value SECONDWIN",
        "USEFUL",
    ]
    llm = _ScriptedLLM(replies)
    agent = SelfRAGAgent(
        retriever,
        AnswerGenerator(cfg=CONFIG, llm=llm),
        CONFIG,
        SelfRAGConfig(max_evidence_refinements=2, max_llm_calls=8),
    )
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert retriever.queries == ["What was Acme revenue in 2015?"]
    assert record.retrieved_chunks[0]["chunk_id"] == "c0"
    assert record.retrieved_chunks[0]["rank"] == 1
    assert record.number_of_evidence_refinements >= 1
    gen_prompts = [p for p in llm.prompts if p.strip().endswith("Answer:") or "\nAnswer:" in p]
    assert gen_prompts
    # Second generation evidence window should include the later slice.
    assert any("SECONDWIN" in p for p in gen_prompts)
    starts = [w["windows"][0]["char_start"] for w in record.evidence_windows]
    assert starts[0] == 0
    assert any(s >= window for s in starts[1:])


def test_rank1_preserved_when_evidence_is_unsupported():
    hit = _rc(chunk_id="keep-rank1", text="hello " + ("z" * 2000))
    retriever = _Retr([hit], [hit])
    replies = ["NEED_RETRIEVAL", "RELEVANT", "UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED", "no", "USEFUL"]
    llm = _ScriptedLLM(replies)
    agent = SelfRAGAgent(
        retriever,
        AnswerGenerator(cfg=CONFIG, llm=llm),
        CONFIG,
        SelfRAGConfig(max_evidence_refinements=2, max_llm_calls=8),
    )
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert record.retrieved_chunks[0]["chunk_id"] == "keep-rank1"
    assert record.retrieved_chunks[0]["rank"] == 1
    assert len(retriever.queries) == 1


def test_max_evidence_refinements_enforced():
    hit = _rc(text="N" * 5000)
    retriever = _Retr([hit], [hit])
    replies = ["NEED_RETRIEVAL", "RELEVANT"] + ["UNSUPPORTED"] * 6 + ["ans", "USEFUL"]
    llm = _ScriptedLLM(replies)
    rag = SelfRAGConfig(max_evidence_refinements=2, max_llm_calls=20)
    agent = SelfRAGAgent(retriever, AnswerGenerator(cfg=CONFIG, llm=llm), CONFIG, rag)
    record = agent.answer("What was Acme revenue in 2015?", "q1")
    assert record.number_of_evidence_refinements <= 2
    assert len(retriever.queries) == 1


def test_baseline_prompt_builder_still_truncates_from_start_only():
    text = ("HEAD" * 200) + ("TAIL" * 200)
    prompt = build_prompt("q", [_rc(text=text)], CONFIG)
    assert "HEAD" in prompt
    # Baseline generator is unchanged: it does not slide to TAIL by itself.
    window = text[: CONFIG.max_chars_per_chunk_in_prompt]
    assert window in prompt or window[:100] in prompt
