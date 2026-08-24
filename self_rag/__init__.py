"""Self-RAG controller layered on the frozen Agentic RAG baseline.

The baseline pipeline (retrieval, prompt, generator, evaluator, Config)
is not modified. This package wraps those components with four optional
reflection stages.
"""

from self_rag.config import SelfRAGConfig
from self_rag.controller import SelfRAGAgent, SelfRAGRecord
from self_rag.labels import (
    EVIDENCE_SUPPORT_LABELS,
    RETRIEVAL_DECISION_LABELS,
    RETRIEVAL_RELEVANCE_LABELS,
    USEFULNESS_LABELS,
    parse_label,
)

__all__ = [
    "SelfRAGConfig",
    "SelfRAGAgent",
    "SelfRAGRecord",
    "parse_label",
    "RETRIEVAL_DECISION_LABELS",
    "RETRIEVAL_RELEVANCE_LABELS",
    "EVIDENCE_SUPPORT_LABELS",
    "USEFULNESS_LABELS",
]
