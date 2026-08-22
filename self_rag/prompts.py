"""Short structured reflection prompts. No chain-of-thought. No gold fields."""

from __future__ import annotations

from typing import Sequence

from config import Config
from generation.answer_generator import format_evidence
from retrieval.hybrid_retriever import RetrievedChunk

DECISION_PROMPT = """Reply with exactly one label. No explanation.
NEED_RETRIEVAL if answering requires documents, tables, or figures.
NO_RETRIEVAL if the question can be answered without retrieved evidence.
Question: {question}
Label:"""

RELEVANCE_PROMPT = """Reply with exactly one label. No explanation.
RELEVANT if any retrieved chunk could help answer the question.
IRRELEVANT if none of the chunks are about the question.
Question: {question}
Evidence:
{evidence}
Label:"""

SUPPORT_PROMPT = """Reply with exactly one label. No explanation.
SUPPORTED if the evidence contains the values needed to answer (including values that must be computed from a table).
UNSUPPORTED if the evidence is missing those values.
Question: {question}
Evidence:
{evidence}
Label:"""

USEFULNESS_PROMPT = """Reply with exactly one label. No explanation.
USEFUL if the draft answers the question using the evidence (a number, short phrase, or a justified insufficient-evidence statement when the evidence truly lacks the values).
NOT_USEFUL if the draft is empty, off-topic, contradicts the evidence, or says insufficient when the evidence actually contains the values.
Question: {question}
Evidence:
{evidence}
Draft:
{draft}
Label:"""


def evidence_block(chunks: Sequence[RetrievedChunk], cfg: Config) -> str:
    limited = list(chunks)[: cfg.max_evidence_chunks]
    return format_evidence(limited, cfg.max_chars_per_chunk_in_prompt)


def retrieval_decision_prompt(question: str) -> str:
    return DECISION_PROMPT.format(question=question)


def retrieval_relevance_prompt(question: str, chunks: Sequence[RetrievedChunk], cfg: Config) -> str:
    return RELEVANCE_PROMPT.format(question=question, evidence=evidence_block(chunks, cfg))


def evidence_support_prompt(question: str, chunks: Sequence[RetrievedChunk], cfg: Config) -> str:
    return SUPPORT_PROMPT.format(question=question, evidence=evidence_block(chunks, cfg))


def answer_usefulness_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    draft: str,
    cfg: Config,
) -> str:
    return USEFULNESS_PROMPT.format(
        question=question,
        evidence=evidence_block(chunks, cfg),
        draft=(draft or "").strip() or "(empty)",
    )


def revision_prompt(question: str, chunks: Sequence[RetrievedChunk], draft: str, cfg: Config) -> str:
    """Controlled revision using the same evidence budget; does not change the baseline generator."""
    return (
        "A previous draft was judged NOT_USEFUL. Using ONLY the evidence, give a concise "
        "corrected final answer. Extract table numbers and compute if needed. "
        "Say exactly \"The retrieved evidence is insufficient to answer this question\" "
        "ONLY if the chunks truly lack the required information.\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence_block(chunks, cfg)}\n"
        f"Previous draft: {(draft or '').strip() or '(empty)'}\n"
        "Answer:"
    )
