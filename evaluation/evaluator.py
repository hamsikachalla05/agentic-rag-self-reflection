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

    def evaluate_log_file(
        self,
        log_path: Path | str,
        questions: list[dict[str, Any]],
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        log_path = Path(log_path)
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gold_by_id = {q["question_id"]: q.get("gold") for q in questions}
        per_q = []
        latencies = []
        tokens = []
        for record in records:
            latencies.append(record.get("latency_seconds") or 0.0)
            tokens.append(record.get("total_tokens") or 0)
            metrics = self.retrieval_metrics(record, gold_by_id.get(record["question_id"]))
            if metrics:
                per_q.append(metrics)
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
            "answer_correctness": "not_implemented",
            "evidence_support": "not_implemented",
            "per_question": [asdict(m) for m in per_q],
        }
        if output_path:
            Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
