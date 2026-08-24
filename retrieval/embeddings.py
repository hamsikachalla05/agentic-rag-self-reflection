"""Open-source embedding model wrapper. No paid embedding APIs."""

from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

from config import CONFIG, Config


class EmbeddingModel:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        self.model = SentenceTransformer(
            self.cfg.embedding_model,
            device=self.cfg.embedding_device,
        )

    def embed_passages(self, texts: Sequence[str]):
        prefixed = [f"{self.cfg.embedding_passage_prefix}{text}" for text in texts]
        return self.model.encode(
            prefixed,
            batch_size=self.cfg.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.embedding_normalize,
            show_progress_bar=len(prefixed) > 16,
        )

    def embed_query(self, query: str):
        text = f"{self.cfg.embedding_query_prefix}{query}"
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.embedding_normalize,
            show_progress_bar=False,
        )[0]
