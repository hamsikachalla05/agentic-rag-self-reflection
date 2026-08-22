"""Normalize T²-RAGBench records. Gold fields are evaluation-only.

Documents: page-level PDFs live in the Hugging Face dataset *repository*
(`data/{FinQA|ConvFinQA|TAT-DQA}/...`), not inside the Arrow tables.
This loader never injects gold answers or gold context into retrieval/LLM paths.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download, list_repo_files

from config import PROJECT_ROOT


HF_REPO = "G4KMU/t2-ragbench"
CONFIGS = ("FinQA", "ConvFinQA", "TAT-DQA")
DEFAULT_SPLITS = {
    "FinQA": "test",
    "ConvFinQA": "turn_0",
    "TAT-DQA": "test",
}

T2_ROOT = PROJECT_ROOT / "data" / "t2_ragbench"
RAW_DIR = T2_ROOT / "raw"
DOC_DIR = T2_ROOT / "documents"
QUESTIONS_DIR = T2_ROOT / "questions"
METADATA_DIR = T2_ROOT / "metadata"


@dataclass
class NormalizedExample:
    question_id: str
    question: str
    gold_answer: str | None
    gold_evidence: str | None
    document_id: str | None
    source: str | None
    page: int | str | None
    dataset: str
    company: str | None = None
    year: int | str | None = None
    split: str | None = None
    original_answer: str | None = None
    hf_pdf_path: str | None = None
    file_name: str | None = None
    context_id: str | None = None

    def to_question_record(self) -> dict[str, Any]:
        """Question file record. `gold` is for the evaluator, not the agent."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "dataset": self.dataset,
            "split": self.split,
            "company": self.company,
            "year": self.year,
            "gold_answer": self.gold_answer,
            "gold_evidence_reference": self.evidence_reference(),
            "gold": {
                "document_id": self.document_id,
                "document_ids": [self.document_id] if self.document_id else [],
                "answer": self.gold_answer,
                "original_answer": self.original_answer,
                "evidence": self.gold_evidence,
                "source_page": self.page,
                "file_name": self.file_name,
            },
        }

    def evidence_reference(self) -> str:
        parts = [self.dataset]
        if self.file_name:
            parts.append(self.file_name)
        if self.page not in (None, ""):
            parts.append(f"source_page={self.page}")
        if self.context_id:
            parts.append(f"context_id={self.context_id}")
        return " | ".join(parts)

    def sidecar(self) -> dict[str, Any]:
        year = self.year
        if year not in (None, ""):
            try:
                year = int(year)
            except (TypeError, ValueError):
                pass
        return {
            "document_id": self.document_id,
            "company": self.company,
            "year": year,
            "filing_type": "annual report",
            "dataset": self.dataset,
            "dataset_page": self.page,
            "context_id": self.context_id,
            "file_name": self.file_name,
            "hf_pdf_path": self.hf_pdf_path,
        }


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def resolve_hf_pdf_path(dataset: str, split: str, file_name: str, repo_files: set[str]) -> str | None:
    file_name = (file_name or "").lstrip("/")
    if not file_name:
        return None
    candidates = [
        f"data/{dataset}/{split}/{file_name}",
        f"data/{dataset}/{file_name}",
    ]
    for path in candidates:
        if path in repo_files:
            return path
    return None


def list_t2_repo_files() -> set[str]:
    return set(list_repo_files(HF_REPO, repo_type="dataset"))


def inspect_config(dataset: str) -> dict[str, Any]:
    from datasets import load_dataset

    loaded = load_dataset(HF_REPO, dataset)
    report: dict[str, Any] = {"dataset": dataset, "splits": {}}
    first_split = next(iter(loaded.keys()))
    report["columns"] = list(loaded[first_split].column_names)
    report["features"] = {k: str(v) for k, v in loaded[first_split].features.items()}
    for split_name, subset in loaded.items():
        examples = []
        for i in range(min(1, len(subset))):
            rec = dict(subset[i])
            slim = {}
            for key, value in rec.items():
                text = _stringify(value) or ""
                slim[key] = text if len(text) <= 240 else text[:240] + f"... <chars={len(text)}>"
            examples.append(slim)
        report["splits"][split_name] = {"n": len(subset), "example": examples[0] if examples else None}
    return report


def inspect_all_configs() -> dict[str, Any]:
    return {name: inspect_config(name) for name in CONFIGS}


def normalize_row(dataset: str, row: dict[str, Any], split: str | None = None) -> NormalizedExample:
    page = row.get("page_number")
    year = row.get("report_year")
    gold_answer = _stringify(row.get("program_answer"))
    original = _stringify(row.get("original_answer"))
    evidence = _stringify(row.get("context"))
    file_name = _stringify(row.get("file_name"))
    context_id = _stringify(row.get("context_id"))
    return NormalizedExample(
        question_id=_stringify(row.get("id")) or "",
        question=_stringify(row.get("question")) or "",
        gold_answer=gold_answer,
        gold_evidence=evidence,
        document_id=context_id,
        source=file_name,
        page=page,
        dataset=dataset,
        company=_stringify(row.get("company_name")),
        year=year,
        split=_stringify(row.get("split")) or split,
        original_answer=original,
        file_name=file_name,
        context_id=context_id,
    )


def iter_normalized(
    dataset: str,
    split: str | None = None,
    limit: int | None = None,
) -> Iterable[NormalizedExample]:
    from datasets import load_dataset

    split = split or DEFAULT_SPLITS[dataset]
    subset = load_dataset(HF_REPO, dataset, split=split)
    count = 0
    for row in subset:
        yield normalize_row(dataset, dict(row), split=split)
        count += 1
        if limit is not None and count >= limit:
            break


def select_small_subset(
    n_per_config: int = 5,
    repo_files: set[str] | None = None,
) -> list[NormalizedExample]:
    repo_files = repo_files or list_t2_repo_files()
    selected: list[NormalizedExample] = []
    for dataset in CONFIGS:
        split = DEFAULT_SPLITS[dataset]
        seen_docs: set[str] = set()
        for example in iter_normalized(dataset, split=split):
            if not example.question or not example.document_id:
                continue
            if example.document_id in seen_docs:
                continue
            hf_path = resolve_hf_pdf_path(dataset, split, example.file_name or "", repo_files)
            if not hf_path:
                continue
            example.hf_pdf_path = hf_path
            selected.append(example)
            seen_docs.add(example.document_id)
            if len(seen_docs) >= n_per_config:
                break
        if len(seen_docs) < n_per_config:
            raise RuntimeError(
                f"Could only resolve {len(seen_docs)} PDFs for {dataset}/{split}; need {n_per_config}"
            )
    return selected


def download_pdf(example: NormalizedExample, dest_dir: Path = DOC_DIR) -> Path:
    if not example.hf_pdf_path:
        raise ValueError(f"No HF PDF path for {example.question_id}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        HF_REPO,
        filename=example.hf_pdf_path,
        repo_type="dataset",
    )
    dest = dest_dir / f"{example.document_id}.pdf"
    dest.write_bytes(Path(cached).read_bytes())
    sidecar_path = dest.with_suffix(".json")
    sidecar_path.write_text(json.dumps(example.sidecar(), indent=2), encoding="utf-8")
    return dest


def prepare_small_subset(n_per_config: int = 5) -> dict[str, Any]:
    T2_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    schema = inspect_all_configs()
    (METADATA_DIR / "schema_inspection.json").write_text(
        json.dumps(schema, indent=2, default=str), encoding="utf-8"
    )

    examples = select_small_subset(n_per_config=n_per_config)
    documents = []
    for example in examples:
        pdf_path = download_pdf(example)
        documents.append(
            {
                "question_id": example.question_id,
                "document_id": example.document_id,
                "local_pdf": str(pdf_path),
                "hf_pdf_path": example.hf_pdf_path,
            }
        )

    questions = {
        "description": (
            "T²-RAGBench 15-question prototype subset. The `gold` object is for "
            "evaluation only and must never be passed to the retriever or LLM."
        ),
        "mapping": {
            "document_id": "HuggingFace context_id; one PDF page per id",
            "chunk_page": (
                "pdfplumber page index inside the downloaded single-page PDF "
                "(almost always 1). Dataset `page_number` is the source 10-K page "
                "and is stored as gold.source_page / sidecar dataset_page."
            ),
            "retrieval_hit": (
                "A retrieved chunk is a document hit if chunk.document_id == "
                "gold.document_id (the gold page PDF). Token overlap between "
                "chunk text and gold.evidence is a secondary extraction check "
                "because 512-token windows are not identical to dataset context strings."
            ),
        },
        "questions": [ex.to_question_record() for ex in examples],
    }
    questions_path = QUESTIONS_DIR / "t2_small_15.json"
    questions_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    (METADATA_DIR / "small_subset_documents.json").write_text(
        json.dumps(documents, indent=2), encoding="utf-8"
    )
    return {
        "n": len(examples),
        "questions_path": str(questions_path),
        "pdf_dir": str(DOC_DIR),
        "by_dataset": {
            name: sum(1 for ex in examples if ex.dataset == name) for name in CONFIGS
        },
    }
