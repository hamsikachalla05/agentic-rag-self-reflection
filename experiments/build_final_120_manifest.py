#!/usr/bin/env python3
"""Build the frozen 120-question T²-RAGBench evaluation manifest.

Selection uses only dataset metadata (ids, question text, company, document id,
evidence surface form). It does not read baseline or Self-RAG logs, scores, or
predictions. The 15-question development set is excluded by question_id.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.t2_ragbench_loader import (
    CONFIGS,
    DEFAULT_SPLITS,
    DOC_DIR,
    QUESTIONS_DIR,
    download_pdf,
    iter_normalized,
    list_t2_repo_files,
    resolve_hf_pdf_path,
)

SEED = 20260822
N_PER_DATASET = 40
DEV_QUESTIONS_PATH = QUESTIONS_DIR / "t2_small_15.json"
MANIFEST_PATH = ROOT / "experiments" / "final_120_questions.json"
DOC_DIR_FINAL = ROOT / "data" / "t2_ragbench" / "documents_final120"
SELECTION_META_PATH = ROOT / "data" / "t2_ragbench" / "metadata" / "final_120_selection.json"

PERCENT_RE = re.compile(
    r"\bpercent(?:age)?s?\b|\bratio\b|\bproportion\b|%",
    re.I,
)
CHANGE_RE = re.compile(
    r"\b(?:change|increase|decrease|difference|growth|delta|variance|net change)\b",
    re.I,
)
SUM_RE = re.compile(
    r"\b(?:sum|combined|aggregate)\b|\btotal of\b",
    re.I,
)


def load_dev_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {q["question_id"] for q in payload["questions"]}


def question_family(question: str) -> str:
    if PERCENT_RE.search(question):
        return "percentage"
    if CHANGE_RE.search(question):
        return "change_difference"
    if SUM_RE.search(question):
        return "sum"
    return "lookup_or_other"


def is_table_heavy(evidence: str | None) -> bool:
    text = evidence or ""
    return text.count("|") >= 8 or "|---" in text or "|_|" in text


def is_valid(example) -> bool:
    if not example.question_id or not example.question.strip():
        return False
    if example.gold_answer in (None, ""):
        return False
    if not example.document_id:
        return False
    if not example.file_name:
        return False
    if not (example.gold_evidence or "").strip():
        return False
    return True


def select_stratified(pool: list, n: int, rng: random.Random) -> list:
    """Prefer unique documents, then unique companies, then fill remaining slots.

    Within each (question_family, table_heavy) stratum, order is a seeded shuffle.
    """
    groups: dict[tuple[str, bool], list] = defaultdict(list)
    for ex in pool:
        family = question_family(ex.question)
        groups[(family, is_table_heavy(ex.gold_evidence))].append(ex)
    for key in groups:
        groups[key].sort(key=lambda e: e.question_id)
        rng.shuffle(groups[key])

    selected: list = []
    used_ids: set[str] = set()
    used_docs: set[str] = set()
    used_companies: set[str] = set()
    keys = sorted(groups)

    def take(unique_doc: bool, unique_company: bool) -> bool:
        progressed = False
        for key in keys:
            if len(selected) >= n:
                return progressed
            bucket = groups[key]
            for idx, ex in enumerate(bucket):
                if ex.question_id in used_ids:
                    continue
                if unique_doc and ex.document_id in used_docs:
                    continue
                company = (ex.company or "").strip()
                if unique_company and company and company in used_companies:
                    continue
                selected.append(ex)
                used_ids.add(ex.question_id)
                used_docs.add(ex.document_id)
                if company:
                    used_companies.add(company)
                bucket.pop(idx)
                progressed = True
                break
        return progressed

    while len(selected) < n and take(unique_doc=True, unique_company=True):
        pass
    while len(selected) < n and take(unique_doc=True, unique_company=False):
        pass
    while len(selected) < n and take(unique_doc=False, unique_company=False):
        pass
    if len(selected) < n:
        raise RuntimeError(f"Could only select {len(selected)} valid questions; need {n}")
    selected.sort(key=lambda e: e.question_id)
    return selected


def collect_pool(dataset: str, repo_files: set[str], excluded_ids: set[str]) -> tuple[list, dict]:
    split = DEFAULT_SPLITS[dataset]
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    pool = []
    stats = Counter()
    for example in iter_normalized(dataset, split=split):
        stats["raw"] += 1
        if example.question_id in excluded_ids:
            stats["excluded_dev"] += 1
            continue
        if not is_valid(example):
            stats["invalid"] += 1
            continue
        if example.question_id in seen_ids:
            stats["duplicate_id"] += 1
            continue
        qnorm = example.question.strip().lower()
        if qnorm in seen_questions:
            stats["duplicate_question"] += 1
            continue
        hf_path = resolve_hf_pdf_path(dataset, split, example.file_name or "", repo_files)
        if not hf_path:
            stats["missing_pdf"] += 1
            continue
        example.hf_pdf_path = hf_path
        seen_ids.add(example.question_id)
        seen_questions.add(qnorm)
        pool.append(example)
        stats["eligible"] += 1
    return pool, dict(stats)


def print_validation(selected: list, dev_ids: set[str], procedure: str) -> None:
    ids = [ex.question_id for ex in selected]
    by_ds = Counter(ex.dataset for ex in selected)
    docs = {ex.document_id for ex in selected}
    companies = {ex.company for ex in selected if ex.company}
    print("=== FINAL 120 MANIFEST VALIDATION ===")
    print(f"1. Total = {len(selected)}")
    print(f"2. FinQA = {by_ds['FinQA']}")
    print(f"3. ConvFinQA = {by_ds['ConvFinQA']}")
    print(f"4. TAT-DQA = {by_ds['TAT-DQA']}")
    print(f"5. Number of unique question IDs = {len(set(ids))}")
    print(f"6. Unique source documents (context_id) = {len(docs)}")
    print(f"   Unique companies = {len(companies)}")
    print("7. Selection procedure:")
    print(procedure)
    print("8. Confirmation: no baseline/Self-RAG outputs, scores, or correctness labels were used.")
    print("9. Confirmation: 15-question development IDs were excluded.")
    overlap = sorted(set(ids) & dev_ids)
    print(f"   Development overlap = {len(overlap)}")
    print("Complete list of 120 selected question IDs:")
    for qid in ids:
        print(qid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    rng = random.Random(SEED)
    dev_ids = load_dev_ids(DEV_QUESTIONS_PATH)
    repo_files = list_t2_repo_files()

    selected: list = []
    pool_stats = {}
    family_counts = {}
    for dataset in CONFIGS:
        pool, stats = collect_pool(dataset, repo_files, dev_ids)
        pool_stats[dataset] = stats
        chosen = select_stratified(pool, N_PER_DATASET, rng)
        family_counts[dataset] = Counter(
            (question_family(ex.question), "table" if is_table_heavy(ex.gold_evidence) else "text")
            for ex in chosen
        )
        selected.extend(chosen)

    selected.sort(key=lambda e: (e.dataset, e.question_id))
    procedure = (
        f"Seeded stratified sample (seed={SEED}) of {N_PER_DATASET} questions per T² config "
        f"from splits {DEFAULT_SPLITS}. Eligible rows require question, question_id, "
        "program_answer, context_id, file_name with a resolvable HF PDF, and non-empty context. "
        "Excluded: 15-question development IDs, duplicate IDs, duplicate question strings, "
        "malformed rows, missing PDFs. Within each dataset, strata are "
        "(percentage | change_difference | sum | lookup_or_other) × (table-heavy | text-heavy) "
        "from question text and '|' table markup in context. Fill order: unique "
        "document_id+company, then unique document_id, then remaining eligible rows. "
        "No model outputs were consulted."
    )
    print_validation(selected, dev_ids, procedure)

    questions = {
        "description": (
            "Frozen T²-RAGBench 120-question final evaluation set "
            "(40 FinQA test + 40 ConvFinQA turn_0 + 40 TAT-DQA test). "
            "Gold is evaluator-only and must never be passed to retrieval or the LLM. "
            "The 15-question development set is excluded."
        ),
        "selection": {
            "seed": SEED,
            "n_per_dataset": N_PER_DATASET,
            "excluded_development_path": str(DEV_QUESTIONS_PATH),
            "n_excluded_development_ids": len(dev_ids),
            "procedure": procedure,
            "used_model_outputs": False,
            "pool_stats": pool_stats,
            "family_counts": {ds: {str(k): v for k, v in c.items()} for ds, c in family_counts.items()},
        },
        "mapping": {
            "document_id": "HuggingFace context_id; one PDF page per id",
            "retrieval_hit": "chunk.document_id == gold.document_id",
        },
        "questions": [ex.to_question_record() for ex in selected],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    SELECTION_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_META_PATH.write_text(
        json.dumps(
            {
                "manifest_path": str(MANIFEST_PATH),
                "n": len(selected),
                "question_ids": [ex.question_id for ex in selected],
                "selection": questions["selection"],
                "unique_documents": len({ex.document_id for ex in selected}),
                "unique_companies": len({ex.company for ex in selected if ex.company}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {MANIFEST_PATH}")

    if args.skip_download:
        return
    DOC_DIR_FINAL.mkdir(parents=True, exist_ok=True)
    for example in selected:
        download_pdf(example, dest_dir=DOC_DIR_FINAL)
    print(f"Downloaded {len(selected)} page PDFs to {DOC_DIR_FINAL}")


if __name__ == "__main__":
    main()
