"""PDF parsing for financial annual reports.

Preserves page numbers, document identifiers, and table text. Metadata is
taken from a sidecar manifest or from a conservative filename convention.
Nothing is invented when a field is unknown.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader


FILENAME_RE = re.compile(
    r"^(?P<company>.+)_(?P<year>19\d{2}|20\d{2})_(?P<filing_type>.+)$",
    re.IGNORECASE,
)


@dataclass
class ParsedPage:
    page: int
    text: str
    tables: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    document_id: str
    source_filename: str
    source_path: str
    company: str | None
    year: int | None
    filing_type: str | None
    pages: list[ParsedPage]
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_manifest(pdf_path: Path) -> dict[str, Any]:
    for candidate in (
        pdf_path.with_suffix(".json"),
        pdf_path.parent / "manifest.json",
    ):
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if candidate.name == "manifest.json":
            by_name = payload.get("documents") or payload
            if isinstance(by_name, dict) and pdf_path.name in by_name:
                return dict(by_name[pdf_path.name])
            continue
        return payload
    return {}


def infer_metadata_from_filename(path: Path) -> dict[str, Any]:
    stem = path.stem
    match = FILENAME_RE.match(stem)
    if not match:
        return {"document_id": stem, "source_filename": path.name}
    company = match.group("company").replace("-", " ").strip()
    year = int(match.group("year"))
    filing_type = match.group("filing_type").replace("-", " ").strip()
    document_id = f"{stem}".lower()
    return {
        "document_id": document_id,
        "company": company,
        "year": year,
        "filing_type": filing_type,
        "source_filename": path.name,
    }


def _table_to_text(table: list[list[Any]]) -> str:
    rows: list[str] = []
    for raw_row in table:
        cells = [("" if cell is None else str(cell).strip()) for cell in raw_row]
        if any(cells):
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    return "[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]"


def _extract_pages(pdf_path: Path) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            tables: list[str] = []
            try:
                extracted_tables = page.extract_tables() or []
            except Exception:
                extracted_tables = []
            for table in extracted_tables:
                rendered = _table_to_text(table)
                if rendered:
                    tables.append(rendered)
            combined = text
            if tables:
                table_block = "\n\n".join(tables)
                if table_block not in combined:
                    combined = (combined + "\n\n" + table_block).strip()
            pages.append(ParsedPage(page=i, text=combined, tables=tables))
    if pages:
        return pages

    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(ParsedPage(page=i, text=text, tables=[]))
    return pages


def parse_pdf(pdf_path: Path | str) -> ParsedDocument:
    pdf_path = Path(pdf_path)
    inferred = infer_metadata_from_filename(pdf_path)
    manifest = _load_manifest(pdf_path)
    metadata = {**inferred, **manifest}

    document_id = str(metadata.get("document_id") or pdf_path.stem)
    company = metadata.get("company")
    year = metadata.get("year")
    if year is not None:
        year = int(year)
    filing_type = metadata.get("filing_type")

    extra = {
        k: v
        for k, v in metadata.items()
        if k not in {"document_id", "company", "year", "filing_type", "source_filename"}
    }

    return ParsedDocument(
        document_id=document_id,
        source_filename=pdf_path.name,
        source_path=str(pdf_path.resolve()),
        company=company,
        year=year,
        filing_type=filing_type,
        pages=_extract_pages(pdf_path),
        extra_metadata=extra,
    )


def parse_pdf_dir(pdf_dir: Path | str) -> list[ParsedDocument]:
    pdf_dir = Path(pdf_dir)
    documents: list[ParsedDocument] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        documents.append(parse_pdf(path))
    return documents
