"""Hybrid retrieval: dense + BM25 with transparent score combination.

This module is the stable retrieval interface. Later Self-RAG / reflection
should call HybridRetriever rather than reimplementing search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from config import CONFIG, Config
from ingestion.chunker import Chunk
from retrieval.bm25 import BM25Index
from retrieval.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    rank: int
    dense_score: float | None
    bm25_score: float | None
    hybrid_score: float
    page: int
    pages: list[int]
    document_id: str
    company: str | None
    year: int | None
    filing_type: str | None
    source_filename: str
    text: str

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "hybrid_score": self.hybrid_score,
            "page": self.page,
            "pages": list(self.pages),
            "document_id": self.document_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {k: 1.0 if v > 0 else 0.0 for k, v in scores.items()}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        chunks: Iterable[Chunk],
        cfg: Config | None = None,
    ):
        self.cfg = cfg or CONFIG
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.chunks_by_id = {c.chunk_id: c for c in chunks}

    def _apply_filters(self, filters: dict[str, Any] | None) -> set[str] | None:
        if not filters:
            return None
        allowed = {c.chunk_id for c in self.chunks_by_id.values() if c.matches_filters(filters)}
        return allowed

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self.cfg
        dense_top_k = dense_top_k if dense_top_k is not None else cfg.dense_top_k
        bm25_top_k = bm25_top_k if bm25_top_k is not None else cfg.bm25_top_k
        final_top_k = final_top_k if final_top_k is not None else cfg.final_top_k

        allowed = self._apply_filters(filters)
        dense_hits = self.vector_store.search(query, dense_top_k, allowed_chunk_ids=allowed)
        bm25_hits = self.bm25_index.search(query, bm25_top_k, allowed_chunk_ids=allowed)

        dense_scores = {chunk_id: score for chunk_id, score in dense_hits}
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_hits}
        all_ids = list(dict.fromkeys([*dense_scores.keys(), *bm25_scores.keys()]))

        if cfg.hybrid_method == "rrf":
            hybrid_scores = self._rrf_scores(dense_hits, bm25_hits)
        else:
            hybrid_scores = self._weighted_minmax_scores(dense_scores, bm25_scores)

        ranked_ids = sorted(all_ids, key=lambda cid: hybrid_scores.get(cid, 0.0), reverse=True)
        ranked_ids = ranked_ids[:final_top_k]

        results: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            chunk = self.chunks_by_id[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    dense_score=dense_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                    hybrid_score=float(hybrid_scores.get(chunk_id, 0.0)),
                    page=chunk.page,
                    pages=list(chunk.pages),
                    document_id=chunk.document_id,
                    company=chunk.company,
                    year=chunk.year,
                    filing_type=chunk.filing_type,
                    source_filename=chunk.source_filename,
                    text=chunk.text,
                )
            )
        return results

    def _weighted_minmax_scores(
        self,
        dense_scores: dict[str, float],
        bm25_scores: dict[str, float],
    ) -> dict[str, float]:
        dense_norm = _minmax(dense_scores)
        bm25_norm = _minmax(bm25_scores)
        ids = set(dense_norm) | set(bm25_norm)
        w_d = self.cfg.hybrid_dense_weight
        w_b = self.cfg.hybrid_bm25_weight
        combined: dict[str, float] = {}
        for chunk_id in ids:
            combined[chunk_id] = (
                w_d * dense_norm.get(chunk_id, 0.0) + w_b * bm25_norm.get(chunk_id, 0.0)
            )
        return combined

    def _rrf_scores(
        self,
        dense_hits: list[tuple[str, float]],
        bm25_hits: list[tuple[str, float]],
    ) -> dict[str, float]:
        k = self.cfg.rrf_k
        scores: dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(dense_hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        for rank, (chunk_id, _) in enumerate(bm25_hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return scores
