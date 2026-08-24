"""LLM-driven retrieval-query formulation (Lumer-style Agentic RAG).

This component is used by the baseline before hybrid retrieval. Self-RAG can
later call the same formulator; it is not wired into Self-RAG yet.

The LLM must not answer the question. Gold answers/evidence are never passed
in. The only input is the original question string.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

from config import CONFIG, Config
from generation.llm_client import LLMOutput, build_llm

QUERY_FORMULATION_MAX_TOKENS = 64

QUERY_FORMULATION_PROMPT = """You formulate a retrieval query for hybrid search (dense embeddings + BM25) over financial filings.

Do NOT answer the question.
Do NOT invent companies, years, figures, or facts that are not in the question.
Do NOT include explanations, labels, quotes, or markdown.

From the question, identify:
- key entities (company / organization names)
- year or other time information
- the financial concept being asked about
- important numerical or unit information

Preserve those terms. Produce one concise search query.

Output ONLY the retrieval query on a single line.

Question: {question}
Retrieval query:"""


@dataclass
class QueryFormulationResult:
    original_question: str
    retrieval_query: str
    prompt: str
    raw_output: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    llm_calls: int = 1
    rate_limit_retries: int = 0
    rate_limit_waits: int = 0
    rate_limit_wait_seconds: float = 0.0

    def to_trace(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "retrieval_query": self.retrieval_query,
            "prompt": self.prompt,
            "raw_output": self.raw_output,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.llm_calls,
            "rate_limit_retries": self.rate_limit_retries,
            "rate_limit_waits": self.rate_limit_waits,
            "rate_limit_wait_seconds": self.rate_limit_wait_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_query_formulation_prompt(question: str) -> str:
    return QUERY_FORMULATION_PROMPT.format(question=question.strip())


def parse_retrieval_query(raw_output: str, fallback_question: str) -> str:
    """Take the first non-empty line as the query; never invent extra terms."""
    text = (raw_output or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    for line in text.splitlines():
        line = line.strip().strip("`")
        line = line.strip('"').strip("'")
        lowered = line.lower()
        for prefix in ("retrieval query:", "query:", "search query:"):
            if lowered.startswith(prefix):
                line = line.split(":", 1)[1].strip().strip('"').strip("'")
                break
        line = " ".join(line.split())
        if line:
            return line
    return " ".join(fallback_question.strip().split())


class QueryFormulator:
    """One LLM call: question in, retrieval query out."""

    def __init__(self, llm=None, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        self.llm = llm or build_llm(self.cfg)

    def formulate(self, question: str) -> QueryFormulationResult:
        prompt = build_query_formulation_prompt(question)
        output = self._generate(prompt)
        raw = getattr(output, "raw", None) or {}
        retrieval_query = parse_retrieval_query(output.text, question)
        return QueryFormulationResult(
            original_question=question,
            retrieval_query=retrieval_query,
            prompt=prompt,
            raw_output=output.text,
            latency_seconds=float(output.latency_seconds or 0.0),
            input_tokens=int(output.input_tokens or 0),
            output_tokens=int(output.output_tokens or 0),
            llm_calls=1,
            rate_limit_retries=int(raw.get("rate_limit_retries") or 0),
            rate_limit_waits=int(raw.get("rate_limit_waits") or 0),
            rate_limit_wait_seconds=float(raw.get("rate_limit_wait_seconds") or 0.0),
        )

    def _generate(self, prompt: str) -> LLMOutput:
        generate = self.llm.generate
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(generate).parameters
            if "max_tokens" in params:
                kwargs["max_tokens"] = QUERY_FORMULATION_MAX_TOKENS
        except (TypeError, ValueError):
            pass
        if kwargs:
            return generate(prompt, **kwargs)
        return generate(prompt)
