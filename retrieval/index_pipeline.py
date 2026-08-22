"""Build and load the frozen baseline retrieval indexes."""

from __future__ import annotations

import json
from pathlib import Path

from config import CONFIG, Config
from ingestion.chunker import chunk_documents
from ingestion.pdf_parser import parse_pdf_dir
from retrieval.bm25 import BM25Index
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import VectorStore


def build_indexes(cfg: Config | None = None, pdf_dir: Path | None = None) -> HybridRetriever:
    cfg = cfg or CONFIG
    pdf_dir = Path(pdf_dir) if pdf_dir is not None else cfg.pdf_dir
    documents = parse_pdf_dir(pdf_dir)
    chunks = chunk_documents(
        documents,
        chunk_size_tokens=cfg.chunk_size_tokens,
        chunk_overlap_tokens=cfg.chunk_overlap_tokens,
        encoding_name=cfg.chunk_tokenizer,
    )
    if not chunks:
        raise RuntimeError(f"No chunks produced from PDFs in {pdf_dir}")

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    embedder = EmbeddingModel(cfg)
    vector_store = VectorStore(embedder, cfg.index_dir)
    vector_store.build(chunks)
    vector_store.save()

    bm25 = BM25Index(cfg.index_dir)
    bm25.build(chunks)
    bm25.save()

    meta = {
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "embedding_model": cfg.embedding_model,
        "chunk_size_tokens": cfg.chunk_size_tokens,
        "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
        "documents": [
            {
                "document_id": d.document_id,
                "company": d.company,
                "year": d.year,
                "filing_type": d.filing_type,
                "n_pages": len(d.pages),
                "source_filename": d.source_filename,
            }
            for d in documents
        ],
    }
    (cfg.index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return HybridRetriever(vector_store, bm25, chunks, cfg)


def load_indexes(cfg: Config | None = None) -> HybridRetriever:
    cfg = cfg or CONFIG
    embedder = EmbeddingModel(cfg)
    vector_store = VectorStore(embedder, cfg.index_dir)
    vector_store.load()
    bm25 = BM25Index(cfg.index_dir)
    bm25.load(vector_store.chunks)
    return HybridRetriever(vector_store, bm25, vector_store.chunks, cfg)
