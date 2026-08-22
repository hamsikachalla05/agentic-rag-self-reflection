"""Slide 700-character evidence windows inside already-retrieved documents.

Does not call FAISS/BM25/hybrid again and does not reorder the original
retrieval list used for Recall@K. Baseline ``format_evidence`` still truncates
each *view* to ``max_chars_per_chunk_in_prompt``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from retrieval.hybrid_retriever import RetrievedChunk


def _chunk_sort_key(chunk_id: str) -> tuple:
    return tuple(chunk_id.split("::"))


@dataclass
class WindowSlice:
    chunk_id: str
    document_id: str
    rank: int
    char_start: int
    char_end: int
    source_chunk_id: str
    reason: str

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_chunk_id": self.source_chunk_id,
            "reason": self.reason,
        }


class EvidenceWindowTracker:
    """Per-chunk character offsets over the frozen retrieval set.

    When a retrieved chunk is exhausted, the next unused index chunk with the
    same ``document_id`` (already produced by the frozen chunker) can fill that
    slot. Original retrieval ranks/ids for evaluation stay on ``original``.
    """

    def __init__(
        self,
        chunks: Sequence[RetrievedChunk],
        window_chars: int,
        chunks_by_id: dict[str, Any] | None = None,
        max_evidence_chunks: int = 5,
    ):
        self.original = list(chunks)
        self.window_chars = max(int(window_chars), 1)
        self.chunks_by_id = chunks_by_id or {}
        self.max_evidence_chunks = max_evidence_chunks
        self.offsets: dict[str, int] = {c.chunk_id: 0 for c in self.original}
        self.sources: dict[str, Any] = {c.chunk_id: c for c in self.original}
        self.history: list[dict[str, Any]] = []
        self._record("initial")

    def _text_of(self, slot_id: str) -> str:
        src = self.sources[slot_id]
        return src.text or ""

    def snapshot(self, reason: str) -> list[dict[str, Any]]:
        rows = []
        for chunk in self.original[: self.max_evidence_chunks]:
            start = self.offsets[chunk.chunk_id]
            src = self.sources[chunk.chunk_id]
            text = src.text or ""
            end = min(start + self.window_chars, len(text))
            rows.append(
                WindowSlice(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    rank=chunk.rank,
                    char_start=start,
                    char_end=end,
                    source_chunk_id=getattr(src, "chunk_id", chunk.chunk_id),
                    reason=reason,
                ).to_log_dict()
            )
        return rows

    def _record(self, reason: str) -> None:
        self.history.append({"reason": reason, "windows": self.snapshot(reason)})

    def current_view(self) -> list[RetrievedChunk]:
        views: list[RetrievedChunk] = []
        for chunk in self.original[: self.max_evidence_chunks]:
            start = self.offsets[chunk.chunk_id]
            src = self.sources[chunk.chunk_id]
            text = (src.text or "")[start : start + self.window_chars]
            views.append(replace(chunk, text=text))
        return views

    def _primary_doc(self) -> str | None:
        if not self.original:
            return None
        return self.original[0].document_id

    def _unused_siblings(self) -> list[Any]:
        doc = self._primary_doc()
        if not doc:
            return []
        used = {getattr(src, "chunk_id", slot) for slot, src in self.sources.items()}
        used.update(self.offsets)
        found = []
        for chunk in self.chunks_by_id.values():
            if getattr(chunk, "document_id", None) != doc:
                continue
            if chunk.chunk_id in used:
                continue
            found.append(chunk)
        found.sort(key=lambda c: _chunk_sort_key(c.chunk_id))
        return found

    def advance(self) -> bool:
        """Move to a later 700-char slice (or same-document sibling). False if none."""
        if not self.original:
            return False
        primary = self._primary_doc()
        moved = False
        for chunk in self.original:
            if chunk.document_id != primary:
                continue
            nxt = self.offsets[chunk.chunk_id] + self.window_chars
            if nxt < len(self._text_of(chunk.chunk_id)):
                self.offsets[chunk.chunk_id] = nxt
                moved = True
        if moved:
            self._record("slide_later_chars")
            return True

        siblings = self._unused_siblings()
        if not siblings:
            return False
        sibling = siblings[0]
        # Fill the rank-1 slot with the next same-document chunk at offset 0.
        primary_slot = self.original[0].chunk_id
        self.sources[primary_slot] = sibling
        self.offsets[primary_slot] = 0
        self._record("same_document_sibling_chunk")
        return True

    def view_fingerprint(self) -> tuple:
        return tuple(
            (
                getattr(self.sources[c.chunk_id], "chunk_id", c.chunk_id),
                self.offsets[c.chunk_id],
            )
            for c in self.original[: self.max_evidence_chunks]
        )
