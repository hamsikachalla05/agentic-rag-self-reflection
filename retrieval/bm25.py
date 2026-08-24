"""BM25 keyword retrieval over the same chunk corpus as dense search."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

from ingestion.chunker import Chunk


TOKEN_RE = re.compile(r"[A-Za-z0-9$%.]+")


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self._id_to_pos: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    @property
    def pickle_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    def build(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._tokenized = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def save(self) -> None:
        payload = {
            "chunk_ids": [c.chunk_id for c in self.chunks],
            "tokenized": self._tokenized,
        }
        with self.pickle_path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        if self.pickle_path.exists():
            with self.pickle_path.open("rb") as handle:
                payload = pickle.load(handle)
            saved_ids = payload.get("chunk_ids")
            current_ids = [c.chunk_id for c in self.chunks]
            if saved_ids == current_ids:
                self._tokenized = payload["tokenized"]
                self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
                return
        self.build(self.chunks)

    def search(
        self,
        query: str,
        top_k: int,
        allowed_chunk_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        allowed = None if allowed_chunk_ids is None else set(allowed_chunk_ids)
        ranked: list[tuple[str, float]] = []
        for i, score in enumerate(scores):
            chunk_id = self.chunks[i].chunk_id
            if allowed is not None and chunk_id not in allowed:
                continue
            ranked.append((chunk_id, float(score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
