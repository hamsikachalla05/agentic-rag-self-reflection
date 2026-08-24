"""Four Self-RAG reflection stages as explicit, independently skippable calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from config import CONFIG, Config
from generation.llm_client import LLMOutput
from retrieval.hybrid_retriever import RetrievedChunk
from self_rag.config import SelfRAGConfig
from self_rag.labels import (
    EVIDENCE_SUPPORT_LABELS,
    RETRIEVAL_DECISION_LABELS,
    RETRIEVAL_RELEVANCE_LABELS,
    USEFULNESS_LABELS,
    parse_followup_query,
    parse_label,
)
from self_rag.prompts import (
    answer_usefulness_prompt,
    evidence_support_prompt,
    retrieval_decision_prompt,
    retrieval_relevance_prompt,
    revision_prompt,
)


@dataclass
class ReflectionResult:
    component: str
    decision: str
    parsed_ok: bool
    prompt: str
    raw_output: str
    followup_query: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    skipped: bool = False

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "decision": self.decision,
            "parsed_ok": self.parsed_ok,
            "skipped": self.skipped,
            "followup_query": self.followup_query,
            "raw_output": self.raw_output,
            "prompt": self.prompt,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
        }


def _call_llm(llm: Any, prompt: str, max_tokens: int) -> LLMOutput:
    generate = llm.generate
    try:
        return generate(prompt, max_tokens=max_tokens)
    except TypeError:
        return generate(prompt)


class Reflector:
    def __init__(self, llm: Any, cfg: Config | None = None, self_rag_cfg: SelfRAGConfig | None = None):
        self.llm = llm
        self.cfg = cfg or CONFIG
        self.self_rag_cfg = self_rag_cfg or SelfRAGConfig()
        self.calls: list[ReflectionResult] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_seconds = 0.0
        self.llm_calls = 0

    def reset_counters(self) -> None:
        self.calls = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_seconds = 0.0
        self.llm_calls = 0

    def remaining_llm_budget(self, already_used: int) -> int:
        return max(0, self.self_rag_cfg.max_llm_calls - already_used)

    def _run(
        self,
        component: str,
        prompt: str,
        labels: tuple[str, ...],
        default: str,
        enabled: bool,
        skip_decision: str,
        allow_followup: bool,
        llm_calls_used: int,
    ) -> ReflectionResult:
        if not enabled:
            result = ReflectionResult(
                component=component,
                decision=skip_decision,
                parsed_ok=True,
                prompt="",
                raw_output="",
                skipped=True,
            )
            self.calls.append(result)
            return result
        if llm_calls_used >= self.self_rag_cfg.max_llm_calls:
            result = ReflectionResult(
                component=component,
                decision=default,
                parsed_ok=False,
                prompt=prompt,
                raw_output="",
                skipped=True,
            )
            self.calls.append(result)
            return result

        output = _call_llm(self.llm, prompt, self.self_rag_cfg.reflection_max_tokens)
        self.llm_calls += 1
        self.input_tokens += int(output.input_tokens or 0)
        self.output_tokens += int(output.output_tokens or 0)
        self.latency_seconds += float(output.latency_seconds or 0.0)
        decision, parsed_ok = parse_label(output.text, labels, default)
        followup = parse_followup_query(output.text) if allow_followup and parsed_ok else None
        result = ReflectionResult(
            component=component,
            decision=decision,
            parsed_ok=parsed_ok,
            prompt=prompt,
            raw_output=output.text or "",
            followup_query=followup,
            input_tokens=int(output.input_tokens or 0),
            output_tokens=int(output.output_tokens or 0),
            latency_seconds=float(output.latency_seconds or 0.0),
        )
        self.calls.append(result)
        return result

    def retrieval_decision(self, question: str, llm_calls_used: int) -> ReflectionResult:
        return self._run(
            component="retrieval_decision",
            prompt=retrieval_decision_prompt(question),
            labels=RETRIEVAL_DECISION_LABELS,
            default="NEED_RETRIEVAL",
            enabled=self.self_rag_cfg.retrieval_decision,
            skip_decision="NEED_RETRIEVAL",
            allow_followup=False,
            llm_calls_used=llm_calls_used,
        )

    def retrieval_relevance(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        llm_calls_used: int,
    ) -> ReflectionResult:
        return self._run(
            component="retrieval_relevance",
            prompt=retrieval_relevance_prompt(question, chunks, self.cfg),
            labels=RETRIEVAL_RELEVANCE_LABELS,
            default="RELEVANT",
            enabled=self.self_rag_cfg.retrieval_relevance,
            skip_decision="RELEVANT",
            allow_followup=False,
            llm_calls_used=llm_calls_used,
        )

    def evidence_support(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        llm_calls_used: int,
    ) -> ReflectionResult:
        return self._run(
            component="evidence_support",
            prompt=evidence_support_prompt(question, chunks, self.cfg),
            labels=EVIDENCE_SUPPORT_LABELS,
            default="SUPPORTED",
            enabled=self.self_rag_cfg.evidence_support,
            skip_decision="SUPPORTED",
            allow_followup=False,
            llm_calls_used=llm_calls_used,
        )

    def answer_usefulness(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        draft: str,
        llm_calls_used: int,
    ) -> ReflectionResult:
        return self._run(
            component="answer_usefulness",
            prompt=answer_usefulness_prompt(question, chunks, draft, self.cfg),
            labels=USEFULNESS_LABELS,
            default="USEFUL",
            enabled=self.self_rag_cfg.answer_usefulness,
            skip_decision="USEFUL",
            allow_followup=False,
            llm_calls_used=llm_calls_used,
        )

    def revise_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        draft: str,
        llm_calls_used: int,
    ) -> LLMOutput | None:
        if llm_calls_used >= self.self_rag_cfg.max_llm_calls:
            return None
        prompt = revision_prompt(question, chunks, draft, self.cfg)
        output = _call_llm(self.llm, prompt, self.cfg.llm_max_new_tokens)
        self.llm_calls += 1
        self.input_tokens += int(output.input_tokens or 0)
        self.output_tokens += int(output.output_tokens or 0)
        self.latency_seconds += float(output.latency_seconds or 0.0)
        self.calls.append(
            ReflectionResult(
                component="answer_revision",
                decision="REVISED",
                parsed_ok=True,
                prompt=prompt,
                raw_output=output.text or "",
                input_tokens=int(output.input_tokens or 0),
                output_tokens=int(output.output_tokens or 0),
                latency_seconds=float(output.latency_seconds or 0.0),
            )
        )
        return output
