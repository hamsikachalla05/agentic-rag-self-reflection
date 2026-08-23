"""Lumer-inspired baseline Agentic RAG controller.

The agent:
  1. receives a question
  2. LLM-formulates a concise retrieval query (does not answer)
  3. extracts metadata filters only when they are explicit in the question
  4. calls the hybrid retriever with the formulated query
  5. passes retrieved evidence to the LLM
  6. returns the final answer

It does not grade retrieval, critique answers, or run Self-RAG.
`formulate_query()` remains a whitespace-only helper so the current Self-RAG
controller is unchanged; Self-RAG can later call QueryFormulator.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from config import CONFIG, Config
from generation.answer_generator import AnswerGenerator, GenerationResult
from retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk

from .query_formulation import QueryFormulator, QueryFormulationResult


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
FILING_PATTERNS = [
    (re.compile(r"\b10-?k\b", re.I), "10-K"),
    (re.compile(r"\b10-?q\b", re.I), "10-Q"),
    (re.compile(r"\b20-?f\b", re.I), "20-F"),
    (re.compile(r"\bannual report\b", re.I), "annual report"),
]


@dataclass
class ExperimentRecord:
    question_id: str
    question: str
    retrieval_query: str
    metadata_filters: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    number_of_retrieval_calls: int
    number_of_llm_calls: int
    retrieval_latency_seconds: float = 0.0
    llm_latency_seconds: float = 0.0
    rate_limit_retries: int = 0
    rate_limit_waits: int = 0
    rate_limit_wait_seconds: float = 0.0
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    original_question: str = ""
    query_formulation_prompt: str = ""
    query_formulation_raw_output: str = ""
    query_formulation_latency_seconds: float = 0.0
    query_formulation_input_tokens: int = 0
    query_formulation_output_tokens: int = 0
    query_formulation_trace: dict[str, Any] = field(default_factory=dict)
    answer_generation_latency_seconds: float = 0.0
    answer_generation_input_tokens: int = 0
    answer_generation_output_tokens: int = 0
    answer_generation_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        cfg: Config | None = None,
        known_companies: list[str] | None = None,
        known_document_ids: list[str] | None = None,
        query_formulator: QueryFormulator | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.cfg = cfg or CONFIG
        self.known_companies = known_companies or self._companies_from_index()
        self.known_document_ids = known_document_ids or self._doc_ids_from_index()
        self.query_formulator = query_formulator or QueryFormulator(
            llm=self.generator.llm,
            cfg=self.cfg,
        )

    def _companies_from_index(self) -> list[str]:
        companies = {
            c.company
            for c in self.retriever.chunks_by_id.values()
            if c.company
        }
        return sorted(companies, key=len, reverse=True)

    def _doc_ids_from_index(self) -> list[str]:
        return sorted({c.document_id for c in self.retriever.chunks_by_id.values()})

    def formulate_query(self, question: str) -> str:
        """Whitespace-only helper kept for the frozen Self-RAG controller."""
        return " ".join(question.strip().split())

    def formulate_retrieval_query(self, question: str) -> QueryFormulationResult:
        """LLM query formulation used by the Lumer-style baseline before retrieval."""
        return self.query_formulator.formulate(question)

    def extract_metadata_filters(self, question: str) -> dict[str, Any]:
        """Extract filters only from explicit question cues. Do not invent values."""
        filters: dict[str, Any] = {}
        years = [int(y) for y in YEAR_RE.findall(question)]
        if len(set(years)) == 1:
            filters["year"] = years[0]

        q_lower = question.lower()
        q_norm = re.sub(r"[^a-z0-9]+", " ", q_lower)
        for company in self.known_companies:
            if not company:
                continue
            company_norm = re.sub(r"[^a-z0-9]+", " ", company.lower()).strip()
            if company_norm and company_norm in q_norm:
                filters["company"] = company
                break

        for pattern, filing_type in FILING_PATTERNS:
            if pattern.search(question):
                filters["filing_type"] = filing_type
                break

        for doc_id in self.known_document_ids:
            if doc_id.lower() in q_lower:
                filters["document_id"] = doc_id
                break

        return filters

    def answer(self, question: str, question_id: str) -> ExperimentRecord:
        t0 = time.perf_counter()
        formulation = self.formulate_retrieval_query(question)
        retrieval_query = formulation.retrieval_query
        filters = self.extract_metadata_filters(question)

        retrieval_calls = 0
        retrieval_latency = 0.0
        chunks: list[RetrievedChunk] = []

        remaining_calls = max(self.cfg.max_retrieval_calls, 1)
        current_filters: dict[str, Any] | None = filters or None

        while remaining_calls > 0:
            remaining_calls -= 1
            retrieval_calls += 1
            t_ret = time.perf_counter()
            chunks = self.retriever.retrieve(retrieval_query, filters=current_filters)
            retrieval_latency += time.perf_counter() - t_ret
            if chunks:
                break
            if (
                current_filters
                and self.cfg.retry_without_filters_if_empty
                and remaining_calls > 0
            ):
                current_filters = None
                continue
            break

        applied_filters = current_filters or {}
        generation: GenerationResult = self.generator.generate(question, chunks)
        latency = time.perf_counter() - t0

        input_tokens = formulation.input_tokens + generation.input_tokens
        output_tokens = formulation.output_tokens + generation.output_tokens
        return ExperimentRecord(
            question_id=question_id,
            question=question,
            retrieval_query=retrieval_query,
            metadata_filters=applied_filters,
            retrieved_chunks=[c.to_log_dict() for c in chunks],
            answer=generation.answer,
            latency_seconds=round(latency, 4),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            number_of_retrieval_calls=retrieval_calls,
            number_of_llm_calls=formulation.llm_calls + generation.llm_calls,
            retrieval_latency_seconds=round(retrieval_latency, 4),
            llm_latency_seconds=round(
                formulation.latency_seconds + generation.latency_seconds, 4
            ),
            rate_limit_retries=formulation.rate_limit_retries + generation.rate_limit_retries,
            rate_limit_waits=formulation.rate_limit_waits + generation.rate_limit_waits,
            rate_limit_wait_seconds=round(
                formulation.rate_limit_wait_seconds + generation.rate_limit_wait_seconds, 4
            ),
            config_snapshot=self.cfg.to_dict(),
            original_question=question,
            query_formulation_prompt=formulation.prompt,
            query_formulation_raw_output=formulation.raw_output,
            query_formulation_latency_seconds=round(formulation.latency_seconds, 4),
            query_formulation_input_tokens=formulation.input_tokens,
            query_formulation_output_tokens=formulation.output_tokens,
            query_formulation_trace=formulation.to_trace(),
            answer_generation_latency_seconds=round(generation.latency_seconds, 4),
            answer_generation_input_tokens=generation.input_tokens,
            answer_generation_output_tokens=generation.output_tokens,
            answer_generation_prompt=generation.prompt,
        )
