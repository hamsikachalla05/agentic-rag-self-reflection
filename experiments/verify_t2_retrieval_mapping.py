#!/usr/bin/env python3
"""Verify T²-RAGBench gold page PDFs correspond to gold context strings.

Does not change Recall@K. Writes a mapping report for the 15-question subset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.t2_ragbench_loader import DOC_DIR, QUESTIONS_DIR
from evaluation.evidence_mapping import coverage
from ingestion.pdf_parser import parse_pdf


def main() -> None:
    questions = json.loads((QUESTIONS_DIR / "t2_small_15.json").read_text(encoding="utf-8"))["questions"]
    rows = []
    for item in questions:
        doc_id = item["gold"]["document_id"]
        pdf = DOC_DIR / f"{doc_id}.pdf"
        sidecar = json.loads(pdf.with_suffix(".json").read_text(encoding="utf-8"))
        parsed = parse_pdf(pdf)
        pdf_text = "\n".join(p.text for p in parsed.pages)
        gold_ctx = item["gold"].get("evidence") or ""
        rows.append(
            {
                "question_id": item["question_id"],
                "dataset": item["dataset"],
                "gold_document_id": doc_id,
                "parsed_document_id": parsed.document_id,
                "ids_match": parsed.document_id == doc_id,
                "pdf_internal_pages": [p.page for p in parsed.pages],
                "dataset_source_page": item["gold"].get("source_page"),
                "sidecar_dataset_page": sidecar.get("dataset_page"),
                "gold_context_coverage_of_pdf_text": coverage(pdf_text, gold_ctx),
                "n_pdf_chars": len(pdf_text),
                "n_gold_context_chars": len(gold_ctx),
            }
        )
    report = {
        "method": (
            "Recall/MRR hit = retrieved chunk.document_id == gold.document_id "
            "(context_id of the Hugging Face page PDF). Dataset page_number is the "
            "10-K page label and is NOT comparable to pdfplumber's page index "
            "(always 1 for these single-page PDFs). Gold `context` is the same "
            "page's text from the dataset, not a separate snippet ID."
        ),
        "retrieval_eval_valid": all(r["ids_match"] for r in rows),
        "page_level_note": (
            "Page-level matching to dataset page_number would be incorrect if "
            "compared to chunk.page. Mapping is document/page-PDF level."
        ),
        "rows": rows,
    }
    out = QUESTIONS_DIR.parent / "metadata" / "retrieval_mapping_verification.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
