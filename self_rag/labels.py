"""Structured Self-RAG labels and fail-safe parsing (no chain-of-thought)."""

from __future__ import annotations

import re

RETRIEVAL_DECISION_LABELS = ("NEED_RETRIEVAL", "NO_RETRIEVAL")
RETRIEVAL_RELEVANCE_LABELS = ("RELEVANT", "IRRELEVANT")
EVIDENCE_SUPPORT_LABELS = ("SUPPORTED", "UNSUPPORTED")
USEFULNESS_LABELS = ("USEFUL", "NOT_USEFUL")

_SKIPPED = "SKIPPED"


def parse_label(text: str, labels: tuple[str, ...], default: str) -> tuple[str, bool]:
    """Return (label, parsed_ok).

    Malformed or empty model output fails safely: return ``default`` and
    ``parsed_ok=False``. Longer labels are matched first so NOT_USEFUL is
    not confused with USEFUL.
    """
    if default not in labels:
        raise ValueError(f"default {default!r} is not in {labels}")
    if text is None:
        return default, False
    raw = str(text).strip()
    if not raw:
        return default, False

    upper = raw.upper().replace("-", "_")
    compact = re.sub(r"[ \t]+", "_", upper)
    ordered = sorted(labels, key=len, reverse=True)
    hits: list[tuple[int, str]] = []
    for label in ordered:
        pattern = label.replace("_", r"[\s_]+")
        match = re.search(rf"(?<![A-Z0-9_]){pattern}(?![A-Z0-9_])", compact)
        if match:
            hits.append((match.start(), label))
        else:
            match = re.search(rf"(?<![A-Z0-9_]){pattern}(?![A-Z0-9_])", upper)
            if match:
                hits.append((match.start(), label))
    if not hits:
        return default, False
    hits.sort(key=lambda item: item[0])
    return hits[0][1], True


def parse_followup_query(text: str) -> str | None:
    """Optional one-line follow-up query. Never required for a valid decision."""
    if not text:
        return None
    match = re.search(
        r"^(?:QUERY|FOLLOWUP|FOLLOWUP_QUERY|RETRIEVAL_QUERY)\s*:\s*(.+)$",
        str(text),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    query = " ".join(match.group(1).strip().split())
    if len(query) < 4:
        return None
    if query.upper().replace(" ", "_") in {
        *RETRIEVAL_DECISION_LABELS,
        *RETRIEVAL_RELEVANCE_LABELS,
        *EVIDENCE_SUPPORT_LABELS,
        *USEFULNESS_LABELS,
    }:
        return None
    return query
