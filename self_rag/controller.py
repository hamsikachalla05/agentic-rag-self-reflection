"""Self-RAG controller around frozen baseline retrieval + generation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.agent import BaselineAgent, ExperimentRecord
from config import CONFIG, Config
from generation.answer_generator import AnswerGenerator, GenerationResult
from retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from self_rag.config import SelfRAGConfig
from self_rag.reflector import Reflector
from self_rag.windows import EvidenceWindowTracker


@dataclass
class SelfRAGRecord:
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
    number_of_retrieval_iterations: int
    number_of_evidence_refinements: int
    retrieval_decision: str
    retrieval_relevance_decision: str
    evidence_support_decision: str
    answer_usefulness_decision: str
    retrieval_latency_seconds: float = 0.0
    llm_latency_seconds: float = 0.0
    rate_limit_retries: int = 0
    rate_limit_waits: int = 0
    rate_limit_wait_seconds: float = 0.0
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    self_rag_config: dict[str, Any] = field(default_factory=dict)
    reflection_trace: list[dict[str, Any]] = field(default_factory=list)
    retrieval_history: list[dict[str, Any]] = field(default_factory=list)
    evidence_windows: list[dict[str, Any]] = field(default_factory=list)
    extra_llm_call_reasons: list[str] = field(default_factory=list)
    original_question: str = ""
    query_formulation_prompt: str = ""
    query_formulation_raw_output: str = ""
    query_formulation_latency_seconds: float = 0.0
    query_formulation_input_tokens: int = 0
    query_formulation_output_tokens: int = 0
    query_formulation_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_baseline_record(self) -> ExperimentRecord:
        return ExperimentRecord(
            question_id=self.question_id,
            question=self.question,
            retrieval_query=self.retrieval_query,
            metadata_filters=self.metadata_filters,
            retrieved_chunks=self.retrieved_chunks,
            answer=self.answer,
            latency_seconds=self.latency_seconds,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            number_of_retrieval_calls=self.number_of_retrieval_calls,
            number_of_llm_calls=self.number_of_llm_calls,
            retrieval_latency_seconds=self.retrieval_latency_seconds,
            llm_latency_seconds=self.llm_latency_seconds,
            rate_limit_retries=self.rate_limit_retries,
            rate_limit_waits=self.rate_limit_waits,
            rate_limit_wait_seconds=self.rate_limit_wait_seconds,
            config_snapshot=self.config_snapshot,
            original_question=self.original_question or self.question,
            query_formulation_prompt=self.query_formulation_prompt,
            query_formulation_raw_output=self.query_formulation_raw_output,
            query_formulation_latency_seconds=self.query_formulation_latency_seconds,
            query_formulation_input_tokens=self.query_formulation_input_tokens,
            query_formulation_output_tokens=self.query_formulation_output_tokens,
            query_formulation_trace=self.query_formulation_trace,
        )


class SelfRAGAgent:
    """Wraps BaselineAgent helpers + HybridRetriever + AnswerGenerator."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        cfg: Config | None = None,
        self_rag_cfg: SelfRAGConfig | None = None,
        reflector: Reflector | None = None,
    ):
        self.cfg = cfg or CONFIG
        self.self_rag_cfg = self_rag_cfg or SelfRAGConfig()
        self.retriever = retriever
        self.generator = generator
        self.baseline = BaselineAgent(retriever, generator, self.cfg)
        self.reflector = reflector or Reflector(
            llm=generator.llm,
            cfg=self.cfg,
            self_rag_cfg=self.self_rag_cfg,
        )

    def _baseline_retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None,
    ) -> tuple[list[RetrievedChunk], dict[str, Any] | None, int, float, list[dict[str, Any]]]:
        """Same retrieve path as the frozen agent: query as-is, empty-filter retry."""
        retrieval_calls = 0
        retrieval_latency = 0.0
        history: list[dict[str, Any]] = []
        current_filters = filters
        chunks: list[RetrievedChunk] = []
        remaining = max(self.cfg.max_retrieval_calls, 1)
        while remaining > 0:
            remaining -= 1
            retrieval_calls += 1
            t_ret = time.perf_counter()
            chunks = self.retriever.retrieve(query, filters=current_filters)
            retrieval_latency += time.perf_counter() - t_ret
            history.append(
                {
                    "iteration": retrieval_calls,
                    "query": query,
                    "filters": dict(current_filters or {}),
                    "chunk_ids": [c.chunk_id for c in chunks],
                    "reason": "baseline_retrieve",
                }
            )
            if chunks:
                break
            if (
                current_filters
                and self.cfg.retry_without_filters_if_empty
                and remaining > 0
            ):
                current_filters = None
                continue
            break
        return chunks, current_filters, retrieval_calls, retrieval_latency, history

    def answer(self, question: str, question_id: str) -> SelfRAGRecord:
        t0 = time.perf_counter()
        self.reflector.reset_counters()
        rag = self.self_rag_cfg
        extra_reasons: list[str] = []

        formulation = self.baseline.formulate_retrieval_query(question)
        retrieval_query = formulation.retrieval_query
        extra_reasons.append("query_formulation")
        filters = self.baseline.extract_metadata_filters(question)

        gen_input_tokens = 0
        gen_output_tokens = 0
        gen_latency = 0.0
        gen_llm_calls = 0
        rate_limit_retries = 0
        rate_limit_waits = 0
        rate_limit_wait_seconds = 0.0

        def llm_used() -> int:
            """Self-RAG stage calls only. Query formulation is extra and does not
            consume the frozen reflection max_llm_calls budget."""
            return gen_llm_calls + self.reflector.llm_calls

        def reflection_budget_used() -> int:
            reserve = 0 if gen_llm_calls else 1
            return llm_used() + reserve

        # Always retrieve first with the frozen retriever so rank-1 is preserved.
        (
            original_chunks,
            applied_filters,
            retrieval_calls,
            retrieval_latency,
            retrieval_history,
        ) = self._baseline_retrieve(retrieval_query, filters or None)
        retrieval_iterations = 1 if retrieval_calls else 0

        tracker = EvidenceWindowTracker(
            original_chunks,
            window_chars=self.cfg.max_chars_per_chunk_in_prompt,
            chunks_by_id=getattr(self.retriever, "chunks_by_id", None) or {},
            max_evidence_chunks=self.cfg.max_evidence_chunks,
        )
        view = tracker.current_view()

        decision = self.reflector.retrieval_decision(question, reflection_budget_used())
        extra_reasons.append("retrieval_decision")

        last_relevance = "SKIPPED"
        last_support = "SKIPPED"
        last_usefulness = "SKIPPED"
        refinements = 0

        if original_chunks:
            relevance = self.reflector.retrieval_relevance(
                question, view, reflection_budget_used()
            )
            last_relevance = relevance.decision
            extra_reasons.append("retrieval_relevance")
            support = self.reflector.evidence_support(
                question, view, reflection_budget_used()
            )
            last_support = support.decision
            extra_reasons.append("evidence_support")

            needs_window = last_support == "UNSUPPORTED" or last_relevance == "IRRELEVANT"
            while (
                needs_window
                and refinements < rag.max_evidence_refinements
                and reflection_budget_used() < rag.max_llm_calls
            ):
                before = tracker.view_fingerprint()
                if not tracker.advance():
                    extra_reasons.append("evidence_refinement_exhausted")
                    break
                if tracker.view_fingerprint() == before:
                    break
                refinements += 1
                view = tracker.current_view()
                extra_reasons.append(f"evidence_window_refinement_{refinements}")
                support = self.reflector.evidence_support(
                    question, view, reflection_budget_used()
                )
                last_support = support.decision
                extra_reasons.append(f"evidence_support_after_refinement_{refinements}")
                if last_support == "SUPPORTED":
                    last_relevance = "RELEVANT"
                    needs_window = False
                    break
                needs_window = last_support == "UNSUPPORTED"

        generation = self._generate(question, view)
        extra_reasons.append("answer_generation")
        gen_llm_calls += generation.llm_calls
        gen_input_tokens += generation.input_tokens
        gen_output_tokens += generation.output_tokens
        gen_latency += generation.latency_seconds
        rate_limit_retries += generation.rate_limit_retries
        rate_limit_waits += generation.rate_limit_waits
        rate_limit_wait_seconds += generation.rate_limit_wait_seconds
        answer = generation.answer
        answer_attempts = 1

        usefulness = self.reflector.answer_usefulness(
            question, view, answer, llm_used()
        )
        last_usefulness = usefulness.decision
        extra_reasons.append("answer_usefulness")

        while (
            last_usefulness == "NOT_USEFUL"
            and answer_attempts < rag.max_answer_attempts
            and llm_used() < rag.max_llm_calls
        ):
            if (
                refinements < rag.max_evidence_refinements
                and tracker.advance()
            ):
                refinements += 1
                view = tracker.current_view()
                extra_reasons.append(f"evidence_window_refinement_after_not_useful_{refinements}")
                support = self.reflector.evidence_support(
                    question, view, reflection_budget_used()
                )
                last_support = support.decision
                extra_reasons.append("evidence_support_after_not_useful")
                generation = self._generate(question, view)
                extra_reasons.append("answer_generation_after_refinement")
                gen_llm_calls += generation.llm_calls
                gen_input_tokens += generation.input_tokens
                gen_output_tokens += generation.output_tokens
                gen_latency += generation.latency_seconds
                answer = generation.answer
            else:
                revised = self.reflector.revise_answer(question, view, answer, llm_used())
                extra_reasons.append("answer_revision")
                if revised is None:
                    break
                answer = revised.text
                gen_input_tokens += revised.input_tokens
                gen_output_tokens += revised.output_tokens
                gen_latency += revised.latency_seconds
            answer_attempts += 1
            usefulness = self.reflector.answer_usefulness(
                question, view, answer, llm_used()
            )
            last_usefulness = usefulness.decision
            extra_reasons.append("answer_usefulness_after_revision")

        latency = time.perf_counter() - t0
        input_tokens = (
            formulation.input_tokens + gen_input_tokens + self.reflector.input_tokens
        )
        output_tokens = (
            formulation.output_tokens + gen_output_tokens + self.reflector.output_tokens
        )
        return SelfRAGRecord(
            question_id=question_id,
            question=question,
            retrieval_query=retrieval_query,
            metadata_filters=applied_filters or {},
            retrieved_chunks=[c.to_log_dict() for c in original_chunks],
            answer=answer,
            latency_seconds=round(latency, 4),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            number_of_retrieval_calls=retrieval_calls,
            number_of_llm_calls=formulation.llm_calls + llm_used(),
            number_of_retrieval_iterations=retrieval_iterations,
            number_of_evidence_refinements=refinements,
            retrieval_decision=decision.decision,
            retrieval_relevance_decision=last_relevance,
            evidence_support_decision=last_support,
            answer_usefulness_decision=last_usefulness,
            retrieval_latency_seconds=round(retrieval_latency, 4),
            llm_latency_seconds=round(
                formulation.latency_seconds + gen_latency + self.reflector.latency_seconds, 4
            ),
            rate_limit_retries=rate_limit_retries + formulation.rate_limit_retries,
            rate_limit_waits=rate_limit_waits + formulation.rate_limit_waits,
            rate_limit_wait_seconds=round(
                rate_limit_wait_seconds + formulation.rate_limit_wait_seconds, 4
            ),
            config_snapshot=self.cfg.to_dict(),
            self_rag_config=rag.to_dict(),
            reflection_trace=[c.to_log_dict() for c in self.reflector.calls],
            retrieval_history=retrieval_history,
            evidence_windows=list(tracker.history),
            extra_llm_call_reasons=extra_reasons,
            original_question=question,
            query_formulation_prompt=formulation.prompt,
            query_formulation_raw_output=formulation.raw_output,
            query_formulation_latency_seconds=round(formulation.latency_seconds, 4),
            query_formulation_input_tokens=formulation.input_tokens,
            query_formulation_output_tokens=formulation.output_tokens,
            query_formulation_trace=formulation.to_trace(),
        )

    def _generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        return self.generator.generate(question, chunks)
