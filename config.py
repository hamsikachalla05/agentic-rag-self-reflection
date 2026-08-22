"""Single configuration file for the baseline Agentic RAG experiment.

Freeze these settings before large-scale runs. Self-RAG / reflection must be
added on top of this baseline without changing chunking, embeddings, retrieval,
context size, or the LLM unless absolutely necessary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # --- Paths ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    pdf_dir: Path = PROJECT_ROOT / "data" / "reports"
    questions_path: Path = PROJECT_ROOT / "data" / "questions" / "sample_questions.json"
    index_dir: Path = PROJECT_ROOT / "data" / "indexes" / "baseline"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # --- Chunking (frozen for fairness vs later Self-RAG) ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    chunk_tokenizer: str = "cl100k_base"

    # --- Embeddings (free / open-source; do not switch mid-experiment) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_passage_prefix: str = ""
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_normalize: bool = True

    # --- Retrieval ---
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    hybrid_dense_weight: float = 0.6
    hybrid_bm25_weight: float = 0.4
    rrf_k: int = 60
    hybrid_method: str = "weighted_minmax"  # or "rrf"

    # --- LLM (free / local) ---
    llm_provider: str = "huggingface_local"
    llm_model: str = "google/flan-t5-base"
    llm_temperature: float = 0.0
    llm_max_new_tokens: int = 96
    llm_max_input_tokens: int = 512
    llm_device: str = "cpu"

    # --- Agent (baseline only: no reflection / critique) ---
    max_retrieval_calls: int = 2
    retry_without_filters_if_empty: bool = True

    # --- Generation context ---
    max_evidence_chunks: int = 5
    max_chars_per_chunk_in_prompt: int = 400

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for key, value in list(raw.items()):
            if isinstance(value, Path):
                raw[key] = str(value)
        return raw


CONFIG = Config()
