"""Token-based chunking with overlap.

Chunks are built **within a page** so page identifiers stay exact for retrieval
evaluation. Pages longer than the chunk size are split with the configured
overlap. Every chunk keeps document identifiers needed for evaluation.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import tiktoken

from ingestion.pdf_parser import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page: int
    pages: list[int]
    company: str | None
    year: int | None
    filing_type: str | None
    source_filename: str
    text: str
    token_count: int
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def matches_filters(self, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if expected in (None, "", []):
                continue
            actual = getattr(self, key, self.extra_metadata.get(key))
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif str(actual).lower() != str(expected).lower() and actual != expected:
                if key == "year":
                    try:
                        if int(actual) != int(expected):
                            return False
                    except (TypeError, ValueError):
                        return False
                else:
                    return False
        return True


def _stable_chunk_id(document_id: str, page: int, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_id}::p{page}::c{ordinal}::{digest}"


def _get_encoder(encoding_name: str):
    return tiktoken.get_encoding(encoding_name)


def chunk_document(
    document: ParsedDocument,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    encoder = _get_encoder(encoding_name)
    chunks: list[Chunk] = []
    ordinal = 0

    for parsed_page in document.pages:
        page_text = (parsed_page.text or "").strip()
        if not page_text:
            continue
        tokens = encoder.encode(page_text)
        start = 0
        n = len(tokens)
        step = max(chunk_size_tokens - chunk_overlap_tokens, 1)
        while start < n:
            end = min(start + chunk_size_tokens, n)
            token_ids = tokens[start:end]
            text = encoder.decode(token_ids).strip()
            if text:
                page = parsed_page.page
                chunk = Chunk(
                    chunk_id=_stable_chunk_id(document.document_id, page, ordinal, text),
                    document_id=document.document_id,
                    page=page,
                    pages=[page],
                    company=document.company,
                    year=document.year,
                    filing_type=document.filing_type,
                    source_filename=document.source_filename,
                    text=text,
                    token_count=len(token_ids),
                    extra_metadata=dict(document.extra_metadata),
                )
                chunks.append(chunk)
                ordinal += 1
            if end >= n:
                break
            start += step

    return chunks


def chunk_documents(documents: Iterable[ParsedDocument], **kwargs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, **kwargs))
    return chunks
