"""Self-RAG flags and loop caps. Independent of frozen baseline Config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SelfRAGConfig:
    """Full Self-RAG enables all four reflection components.

    Each flag can be turned off for later ablations. The frozen baseline
    `config.CONFIG` object does not include these fields and does not
    read this class.
    """

    retrieval_decision: bool = True
    retrieval_relevance: bool = True
    evidence_support: bool = True
    answer_usefulness: bool = True

    # Loop caps so reflection cannot retrieve/generate forever.
    max_retrieval_iterations: int = 2
    max_answer_attempts: int = 2
    max_llm_calls: int = 12
    reflection_max_tokens: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FULL_SELF_RAG = SelfRAGConfig()
