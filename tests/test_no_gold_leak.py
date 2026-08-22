from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "reports" / "manifest.json"


def test_gold_pages_are_not_embedded_in_agent_prompt_builder():
    """Safety: gold fields live only in the questions file, not in generation prompts."""
    from generation.answer_generator import SYSTEM_INSTRUCTIONS, build_prompt
    from retrieval.hybrid_retriever import RetrievedChunk

    questions = json.loads(
        (ROOT / "data" / "questions" / "sample_questions.json").read_text(encoding="utf-8")
    )
    gold_blob = json.dumps(questions)
    assert "gold" in gold_blob

    chunk = RetrievedChunk(
        chunk_id="x",
        rank=1,
        dense_score=1.0,
        bm25_score=1.0,
        hybrid_score=1.0,
        page=3,
        pages=[3],
        document_id="demo",
        company="Demo",
        year=2023,
        filing_type="annual report",
        source_filename="demo.pdf",
        text="Total revenues were 12.4 billion.",
    )
    prompt = build_prompt("What were revenues?", [chunk], cfg=__import__("config").CONFIG)
    assert "gold" not in prompt.lower()
    assert "12.4 billion" in prompt
    assert "only" in SYSTEM_INSTRUCTIONS.lower()
