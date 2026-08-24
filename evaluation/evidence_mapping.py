"""Map dataset gold page-text onto frozen 512-token chunks.

T²-RAGBench `context` is the gold source page as a string. Our retriever
returns chunks of the *PDF we ingested*, which are not guaranteed to equal
that string (page-local 512-token windows, overlap, pdfplumber vs dataset
tokenization).

A retrieval hit is therefore:

1. Primary: chunk.document_id == gold.document_id
   (each HF PDF is one report page keyed by context_id).
2. Secondary: token coverage of gold.evidence by the chunk text, used to
   verify that extraction actually captured the supporting page.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from retrieval.bm25 import tokenize


def coverage(chunk_text: str, gold_text: str) -> float:
    gold_tokens = set(tokenize(gold_text or ""))
    if not gold_tokens:
        return 0.0
    chunk_tokens = set(tokenize(chunk_text or ""))
    return len(gold_tokens & chunk_tokens) / len(gold_tokens)


def best_overlap(
    retrieved: Iterable[dict[str, Any]],
    gold_evidence: str,
    chunks_by_id: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    best = {"chunk_id": None, "rank": None, "coverage": 0.0}
    for item in retrieved:
        text = item.get("text") or ""
        if not text and chunks_by_id is not None:
            chunk = chunks_by_id.get(item.get("chunk_id"))
            if chunk is not None:
                text = getattr(chunk, "text", "") or ""
        cov = coverage(text, gold_evidence)
        if cov > best["coverage"]:
            best = {
                "chunk_id": item.get("chunk_id"),
                "rank": item.get("rank"),
                "coverage": cov,
            }
    return best
