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

    def answer(self, question: str, question_id: str) -> SelfRAGRecord:
        t0 = time.perf_counter()
        self.reflector.reset_counters()
        rag = self.self_rag_cfg

        retrieval_query = self.baseline.formulate_query(question)
        filters = self.baseline.extract_metadata_filters(question)
        current_filters: dict[str, Any] | None = filters or None
        current_query = retrieval_query

        retrieval_calls = 0
        retrieval_iterations = 0
        retrieval_latency = 0.0
        chunks: list[RetrievedChunk] = []
        retrieval_history: list[dict[str, Any]] = []
        gen_input_tokens = 0
        gen_output_tokens = 0
        gen_latency = 0.0
        gen_llm_calls = 0
        rate_limit_retries = 0
        rate_limit_waits = 0
        rate_limit_wait_seconds = 0.0

        def llm_used() -> int:
            return gen_llm_calls + self.reflector.llm_calls

        def reflection_budget_used() -> int:
            # Keep one call for the first answer so reflection cannot starve generation.
            reserve = 0 if gen_llm_calls else 1
            return llm_used() + reserve

        decision = self.reflector.retrieval_decision(question, reflection_budget_used())
        need_retrieval = decision.decision != "NO_RETRIEVAL"

        last_relevance = "SKIPPED"
        last_support = "SKIPPED"
        last_usefulness = "SKIPPED"
        seen_ids: set[tuple[str, ...]] = set()

        if need_retrieval:
            while retrieval_iterations < rag.max_retrieval_iterations:
                if reflection_budget_used() >= rag.max_llm_calls and retrieval_iterations > 0:
                    break
                retrieval_iterations += 1
                retrieval_calls += 1
                t_ret = time.perf_counter()
                chunks = self.retriever.retrieve(current_query, filters=current_filters)
                retrieval_latency += time.perf_counter() - t_ret
                chunk_key = tuple(c.chunk_id for c in chunks)
                retrieval_history.append(
                    {
                        "iteration": retrieval_iterations,
                        "query": current_query,
                        "filters": dict(current_filters or {}),
                        "chunk_ids": list(chunk_key),
                    }
                )

                if (
                    not chunks
                    and current_filters
                    and self.cfg.retry_without_filters_if_empty
                    and retrieval_iterations < rag.max_retrieval_iterations
                ):
                    current_filters = None
                    continue

                relevance = self.reflector.retrieval_relevance(
                    question, chunks, reflection_budget_used()
                )
                last_relevance = relevance.decision
                support = self.reflector.evidence_support(
                    question, chunks, reflection_budget_used()
                )
                last_support = support.decision

                inadequate = last_relevance == "IRRELEVANT" or last_support == "UNSUPPORTED"
                duplicate = chunk_key in seen_ids
                seen_ids.add(chunk_key)
                if not inadequate or retrieval_iterations >= rag.max_retrieval_iterations:
                    break
                if duplicate and not (relevance.followup_query or support.followup_query):
                    if current_filters:
                        current_filters = None
                        continue
                    break

                followup = relevance.followup_query or support.followup_query
                if followup:
                    current_query = followup
                if current_filters:
                    current_filters = None

        generation = self._generate(question, chunks)
        gen_llm_calls += generation.llm_calls
        gen_input_tokens += generation.input_tokens
        gen_output_tokens += generation.output_tokens
        gen_latency += generation.latency_seconds
        rate_limit_retries += generation.rate_limit_retries
        rate_limit_waits += generation.rate_limit_waits
        rate_limit_wait_seconds += generation.rate_limit_wait_seconds
        answer = generation.answer
        answer_attempts = 1

        usefulness = self.reflector.answer_usefulness(question, chunks, answer, llm_used())
        last_usefulness = usefulness.decision

        while (
            last_usefulness == "NOT_USEFUL"
            and answer_attempts < rag.max_answer_attempts
            and llm_used() < rag.max_llm_calls
        ):
            missing_evidence = not chunks
            can_retrieve_again = retrieval_iterations < rag.max_retrieval_iterations and (
                need_retrieval or missing_evidence
            )
            if can_retrieve_again:
                retrieval_iterations += 1
                retrieval_calls += 1
                if missing_evidence:
                    current_query = retrieval_query
                    current_filters = filters or None
                elif current_filters:
                    current_filters = None
                t_ret = time.perf_counter()
                new_chunks = self.retriever.retrieve(current_query, filters=current_filters)
                retrieval_latency += time.perf_counter() - t_ret
                if new_chunks:
                    chunks = new_chunks
                retrieval_history.append(
                    {
                        "iteration": retrieval_iterations,
                        "query": current_query,
                        "filters": dict(current_filters or {}),
                        "chunk_ids": [c.chunk_id for c in chunks],
                        "reason": "not_useful_retry",
                    }
                )
                generation = self._generate(question, chunks)
                gen_llm_calls += generation.llm_calls
                gen_input_tokens += generation.input_tokens
                gen_output_tokens += generation.output_tokens
                gen_latency += generation.latency_seconds
                answer = generation.answer
            else:
                revised = self.reflector.revise_answer(question, chunks, answer, llm_used())
                if revised is None:
                    break
                answer = revised.text
                gen_input_tokens += revised.input_tokens
                gen_output_tokens += revised.output_tokens
                gen_latency += revised.latency_seconds
            answer_attempts += 1
            usefulness = self.reflector.answer_usefulness(question, chunks, answer, llm_used())
            last_usefulness = usefulness.decision

        latency = time.perf_counter() - t0
        input_tokens = gen_input_tokens + self.reflector.input_tokens
        output_tokens = gen_output_tokens + self.reflector.output_tokens
        return SelfRAGRecord(
            question_id=question_id,
            question=question,
            retrieval_query=retrieval_query,
            metadata_filters=current_filters or {},
            retrieved_chunks=[c.to_log_dict() for c in chunks],
            answer=answer,
            latency_seconds=round(latency, 4),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            number_of_retrieval_calls=retrieval_calls,
            number_of_llm_calls=llm_used(),
            number_of_retrieval_iterations=retrieval_iterations,
            retrieval_decision=decision.decision,
            retrieval_relevance_decision=last_relevance,
            evidence_support_decision=last_support,
            answer_usefulness_decision=last_usefulness,
            retrieval_latency_seconds=round(retrieval_latency, 4),
            llm_latency_seconds=round(gen_latency + self.reflector.latency_seconds, 4),
            rate_limit_retries=rate_limit_retries,
            rate_limit_waits=rate_limit_waits,
            rate_limit_wait_seconds=round(rate_limit_wait_seconds, 4),
            config_snapshot=self.cfg.to_dict(),
            self_rag_config=rag.to_dict(),
            reflection_trace=[c.to_log_dict() for c in self.reflector.calls],
            retrieval_history=retrieval_history,
        )

    def _generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        return self.generator.generate(question, chunks)
