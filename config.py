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

    # --- LLM (frozen local generator; retrieval knobs stay frozen) ---
    llm_provider: str = "openai_compatible"
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    llm_temperature: float = 0.0
    llm_max_new_tokens: int = 512
    llm_max_input_tokens: int = 512  # local HF encoder-decoder only
    llm_device: str = "cpu"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key_env: str = "LLM_API_KEY"
    llm_num_ctx: int = 2048
    llm_think: bool = False

    # --- Agent (baseline only: no reflection / critique) ---
    max_retrieval_calls: int = 2
    retry_without_filters_if_empty: bool = True

    # --- Generation context ---
    max_evidence_chunks: int = 5
    # 700 chars/chunk: 5 retrieved chunks + instructions + longest 15-Q prompt
    # is ~1430 input tokens (cl100k). With llm_max_new_tokens=512 this stays
    # under llm_num_ctx=2048. 800 chars overflows that budget.
    max_chars_per_chunk_in_prompt: int = 700

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for key, value in list(raw.items()):
            if isinstance(value, Path):
                raw[key] = str(value)
        return raw


CONFIG = Config()
