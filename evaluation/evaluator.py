"""Evaluation is separate from retrieval.

Ground-truth answers / gold evidence are used ONLY here. The agent and
generator never receive gold context.

Currently implemented:
  - Recall@K over gold document/page identifiers
  - MRR over gold document/page identifiers
  - latency and token-usage aggregates from experiment logs

Configurable / stubbed for later:
  - answer correctness
  - evidence support
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from evaluation.answer_metrics import is_answer_correct
from evaluation.evidence_mapping import best_overlap


@dataclass
class RetrievalMetrics:
    question_id: str
    recall_at_k: dict[int, float]
    mrr: float
    hit: bool
    gold_pages: list[int]
    gold_document_ids: list[str]
    retrieved_pages: list[int]


class Evaluator:
    def __init__(self, ks: Iterable[int] | None = None):
        self.ks = list(ks) if ks is not None else [1, 3, 5, 8, 10]

    def retrieval_metrics(
        self,
        record: dict[str, Any],
        gold: dict[str, Any] | None,
    ) -> RetrievalMetrics | None:
        if not gold:
            return None
        gold_pages = [int(p) for p in gold.get("pages") or gold.get("gold_pages") or []]
        gold_docs = [
            str(d)
            for d in gold.get("document_ids")
            or ([gold["document_id"]] if gold.get("document_id") else [])
        ]
        retrieved = record.get("retrieved_chunks") or []
        retrieved_pages = []
        for c in retrieved:
            if c.get("pages"):
                retrieved_pages.extend(int(p) for p in c["pages"])
            elif c.get("page") is not None:
                retrieved_pages.append(int(c["page"]))

        def is_hit(chunk: dict[str, Any]) -> bool:
            doc_ok = True
            if gold_docs:
                doc_ok = str(chunk.get("document_id")) in gold_docs
            page_ok = True
            if gold_pages:
                chunk_pages = {int(p) for p in chunk.get("pages") or []}
                if chunk.get("page") is not None:
                    chunk_pages.add(int(chunk["page"]))
                page_ok = bool(chunk_pages & set(gold_pages))
            return doc_ok and page_ok

        hits = [is_hit(c) for c in retrieved]
        recall = {}
        for k in self.ks:
            window = hits[:k]
            recall[k] = 1.0 if any(window) else 0.0
        mrr = 0.0
        for i, hit in enumerate(hits, start=1):
            if hit:
                mrr = 1.0 / i
                break
        return RetrievalMetrics(
            question_id=record.get("question_id", ""),
            recall_at_k=recall,
            mrr=mrr,
            hit=any(hits),
            gold_pages=gold_pages,
            gold_document_ids=gold_docs,
            retrieved_pages=retrieved_pages,
        )

    def summarize(self, per_question: list[RetrievalMetrics]) -> dict[str, Any]:
        if not per_question:
            return {"n": 0}
        n = len(per_question)
        summary: dict[str, Any] = {"n": n, "mrr": sum(m.mrr for m in per_question) / n}
        keys = sorted({k for m in per_question for k in m.recall_at_k})
        summary["recall_at_k"] = {
            k: sum(m.recall_at_k.get(k, 0.0) for m in per_question) / n for k in keys
        }
        summary["hit_rate"] = sum(1.0 for m in per_question if m.hit) / n
        return summary

    def _by_dataset(
        self,
        per_q: list[RetrievalMetrics],
        correctness: list[dict[str, Any]],
        records: list[dict[str, Any]],
        dataset_by_id: dict[str, Any],
    ) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        metrics_by_id = {m.question_id: m for m in per_q}
        for record in records:
            qid = record.get("question_id")
            dataset = str(record.get("dataset") or dataset_by_id.get(qid) or "unknown")
            bucket = groups.setdefault(
                dataset,
                {"retrieval": [], "correctness": [], "latencies": [], "tokens": []},
            )
            if qid in metrics_by_id:
                bucket["retrieval"].append(metrics_by_id[qid])
            bucket["latencies"].append(record.get("latency_seconds") or 0.0)
            bucket["tokens"].append(record.get("total_tokens") or 0)
        for row in correctness:
            dataset = str(row.get("dataset") or "unknown")
            groups.setdefault(
                dataset,
                {"retrieval": [], "correctness": [], "latencies": [], "tokens": []},
            )["correctness"].append(row["correct"])
        out: dict[str, Any] = {}
        for dataset, bucket in groups.items():
            n_corr = bucket["correctness"]
            out[dataset] = {
                "retrieval": self.summarize(bucket["retrieval"]),
                "answer_correctness": {
                    "accuracy": (sum(n_corr) / len(n_corr)) if n_corr else None,
                    "n": len(n_corr),
                    "n_correct": sum(n_corr) if n_corr else 0,
                },
                "latency": {
                    "mean_seconds": sum(bucket["latencies"]) / len(bucket["latencies"])
                    if bucket["latencies"]
                    else None
                },
                "tokens": {
                    "mean_total": sum(bucket["tokens"]) / len(bucket["tokens"])
                    if bucket["tokens"]
                    else None
                },
            }
        return out

    def evaluate_log_file(
        self,
        log_path: Path | str,
        questions: list[dict[str, Any]],
        output_path: Path | str | None = None,
        chunks_by_id: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        log_path = Path(log_path)
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gold_by_id = {q["question_id"]: q.get("gold") for q in questions}
        dataset_by_id = {q["question_id"]: q.get("dataset") for q in questions}
        per_q = []
        latencies = []
        tokens = []
        correctness: list[dict[str, Any]] = []
        overlaps: list[dict[str, Any]] = []
        for record in records:
            latencies.append(record.get("latency_seconds") or 0.0)
            tokens.append(record.get("total_tokens") or 0)
            gold = gold_by_id.get(record["question_id"])
            dataset = record.get("dataset") or dataset_by_id.get(record["question_id"])
            metrics = self.retrieval_metrics(record, gold)
            if metrics:
                per_q.append(metrics)
            if gold and (gold.get("answer") not in (None, "") or gold.get("original_answer") not in (None, "")):
                ok = is_answer_correct(
                    record.get("answer") or "",
                    gold.get("answer"),
                    gold.get("original_answer"),
                )
                correctness.append(
                    {
                        "question_id": record.get("question_id"),
                        "dataset": dataset,
                        "correct": ok,
                        "prediction": record.get("answer"),
                        "program_answer": gold.get("answer"),
                        "original_answer": gold.get("original_answer"),
                    }
                )
            if gold and gold.get("evidence"):
                overlaps.append(
                    {
                        "question_id": record.get("question_id"),
                        "dataset": dataset,
                        **best_overlap(
                            record.get("retrieved_chunks") or [],
                            gold.get("evidence") or "",
                            chunks_by_id=chunks_by_id,
                        ),
                    }
                )
        by_dataset = self._by_dataset(per_q, correctness, records, dataset_by_id)
        result = {
            "retrieval": self.summarize(per_q),
            "latency": {
                "mean_seconds": sum(latencies) / len(latencies) if latencies else None,
                "n": len(latencies),
            },
            "tokens": {
                "mean_total": sum(tokens) / len(tokens) if tokens else None,
                "n": len(tokens),
            },
            "answer_correctness": (
                {
                    "accuracy": sum(1 for c in correctness if c["correct"]) / len(correctness),
                    "n": len(correctness),
                    "n_correct": sum(1 for c in correctness if c["correct"]),
                    "gold_field": "program_answer (canonical); original_answer accepted as equivalent surface form",
                    "per_question": correctness,
                }
                if correctness
                else "not_implemented"
            ),
            "by_dataset": by_dataset,
            "evidence_support": "not_implemented",
            "evidence_overlap": overlaps,
            "per_question": [asdict(m) for m in per_q],
        }
        if output_path:
            Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
