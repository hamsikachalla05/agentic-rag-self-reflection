"""Local FAISS vector store over document chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np

from ingestion.chunker import Chunk
from retrieval.embeddings import EmbeddingModel


class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel, index_dir: Path):
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self._id_to_pos: dict[str, int] = {}
        self._index: faiss.Index | None = None

    @property
    def index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    def build(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        texts = [c.text for c in self.chunks]
        if not texts:
            dim = self.embedding_model.model.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(dim)
            return
        embeddings = self.embedding_model.embed_passages(texts)
        vectors = np.asarray(embeddings, dtype="float32")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def save(self) -> None:
        if self._index is None:
            raise RuntimeError("Vector store has not been built")
        faiss.write_index(self._index, str(self.index_path))
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def load(self) -> None:
        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError(f"No vector index found under {self.index_dir}")
        self._index = faiss.read_index(str(self.index_path))
        self.chunks = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                self.chunks.append(Chunk(**payload))
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def search(
        self,
        query: str,
        top_k: int,
        allowed_chunk_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Vector store is not loaded")
        if not self.chunks:
            return []
        query_vec = np.asarray(
            [self.embedding_model.embed_query(query)], dtype="float32"
        )
        allowed = None if allowed_chunk_ids is None else set(allowed_chunk_ids)
        # Over-retrieve then filter so metadata constraints are exact.
        retrieve_k = min(len(self.chunks), max(top_k * 8, top_k))
        scores, indices = self._index.search(query_vec, retrieve_k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if allowed is not None and chunk.chunk_id not in allowed:
                continue
            results.append((chunk.chunk_id, float(score)))
            if len(results) >= top_k:
                break
        return results
