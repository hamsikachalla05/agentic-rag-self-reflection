"""Answer generation from retrieved evidence only.

Ground-truth dataset contexts are never passed here. The LLM sees only the
user question and chunks returned by the retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from config import CONFIG, Config
from generation.llm_client import build_llm
from retrieval.hybrid_retriever import RetrievedChunk


SYSTEM_INSTRUCTIONS = """Answer using ONLY the retrieved evidence.
Do not use outside knowledge. Do not invent figures.
If evidence is insufficient, reply exactly: The retrieved evidence is insufficient to answer this question.
Cite source filename and page. Write 1-3 sentences."""


@dataclass
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    llm_calls: int
    prompt: str
    rate_limit_retries: int = 0
    rate_limit_waits: int = 0
    rate_limit_wait_seconds: float = 0.0


def format_evidence(chunks: Sequence[RetrievedChunk], max_chars_per_chunk: int) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    blocks = []
    for chunk in chunks:
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk].rstrip() + " ..."
        pages = ",".join(str(p) for p in chunk.pages) if chunk.pages else str(chunk.page)
        header = f"[{chunk.rank}] {chunk.source_filename} p.{chunk.page}"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: Sequence[RetrievedChunk], cfg: Config) -> str:
    evidence = format_evidence(chunks, cfg.max_chars_per_chunk_in_prompt)
    # Same evidence-only prompt for every backend (Gemini, FLAN-T5, etc.).
    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence}\n"
        f"Answer:"
    )


class AnswerGenerator:
    def __init__(self, cfg: Config | None = None, llm=None):
        self.cfg = cfg or CONFIG
        self.llm = llm or build_llm(self.cfg)

    def generate(self, question: str, chunks: Sequence[RetrievedChunk]) -> GenerationResult:
        limited = list(chunks)[: self.cfg.max_evidence_chunks]
        prompt = build_prompt(question, limited, self.cfg)
        output = self.llm.generate(prompt)
        raw = getattr(output, "raw", None) or {}
        return GenerationResult(
            answer=output.text,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            total_tokens=output.input_tokens + output.output_tokens,
            latency_seconds=output.latency_seconds,
            llm_calls=1,
            prompt=prompt,
            rate_limit_retries=int(raw.get("rate_limit_retries") or 0),
            rate_limit_waits=int(raw.get("rate_limit_waits") or 0),
            rate_limit_wait_seconds=float(raw.get("rate_limit_wait_seconds") or 0.0),
        )
