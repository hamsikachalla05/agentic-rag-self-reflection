#!/usr/bin/env python3
"""Document T²-RAGBench gold-answer fields. Evaluation-only; no retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from data_loader.t2_ragbench_loader import CONFIGS, DEFAULT_SPLITS, METADATA_DIR, HF_REPO


def slim_context(text: str, n: int = 400) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


def main() -> None:
    report: dict = {
        "note": (
            "T²-RAGBench does not include a program/derivation string. "
            "program_answer is the executed numeric result. original_answer is "
            "the source dataset's surface form. context is the gold page text."
        ),
        "canonical_gold_field": "program_answer",
        "equivalent_surface_field": "original_answer",
        "examples": {},
    }
    for cfg in CONFIGS:
        split = DEFAULT_SPLITS[cfg]
        ds = load_dataset(HF_REPO, cfg, split=split)
        rows = []
        for i in range(min(5, len(ds))):
            rec = ds[i]
            extra = [k for k in rec.keys() if any(s in k.lower() for s in ("program", "deriv", "exe", "gold"))]
            extra = [k for k in extra if k != "program_answer"]
            rows.append(
                {
                    "id": rec.get("id"),
                    "question": rec.get("question"),
                    "program_answer": rec.get("program_answer"),
                    "original_answer": rec.get("original_answer"),
                    "context": slim_context(rec.get("context") or ""),
                    "file_name": rec.get("file_name"),
                    "page_number": rec.get("page_number"),
                    "context_id": rec.get("context_id"),
                    "derivation_fields_present": extra,
                }
            )
        report["examples"][cfg] = {"split": split, "n_printed": len(rows), "rows": rows}
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    path = METADATA_DIR / "gold_answer_field_guide.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
