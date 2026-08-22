#!/usr/bin/env python3
"""Run full Self-RAG on a T²-RAGBench question file.

Reuses frozen baseline retrieval, generator, evaluator, and logging.
Gold answers are attached to logs after generation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from data_loader.t2_ragbench_loader import DOC_DIR, QUESTIONS_DIR, prepare_small_subset
from evaluation.evaluator import Evaluator
from evaluation.logging_utils import ExperimentLogger
from experiments.run_t2_ragbench import (
    agent_question_only,
    assert_no_gold_in_prompt,
    attach_eval_metadata,
    load_questions,
)
from generation.answer_generator import AnswerGenerator
from retrieval.hybrid_retriever import RetrievedChunk
from retrieval.index_pipeline import build_indexes, load_indexes
from self_rag.config import SelfRAGConfig
from self_rag.controller import SelfRAGAgent


GOLD_LEAK_MARKERS = (
    "gold_answer",
    "program_answer",
    "original_answer",
    "gold_evidence",
    "gold_context",
    "gold.evidence",
)


def assert_no_gold_in_self_rag(question: str, item: dict, retrieved_chunks, record) -> None:
    assert_no_gold_in_prompt(question, item, retrieved_chunks)
    blob = json.dumps(
        {
            "question": record.question,
            "answer": record.answer,
            "trace_prompts": [step.get("prompt") or "" for step in record.reflection_trace],
            "windows": record.evidence_windows,
            "reasons": record.extra_llm_call_reasons,
        },
        ensure_ascii=False,
    ).lower()
    for marker in GOLD_LEAK_MARKERS:
        if marker in blob:
            raise AssertionError(f"Gold field {marker!r} leaked into Self-RAG model-facing text")
    gold_answer = (item.get("gold_answer") or "").strip()
    if gold_answer and len(gold_answer) >= 8 and any(ch.isalpha() for ch in gold_answer):
        for step in record.reflection_trace:
            prompt = (step.get("prompt") or "").lower()
            if gold_answer.lower() in prompt:
                raise AssertionError("Gold answer leaked into a reflection prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Self-RAG on T²-RAGBench")
    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_DIR / "t2_small_15.json",
    )
    parser.add_argument("--pdf-dir", type=Path, default=DOC_DIR)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=CONFIG.project_root / "data" / "indexes" / "t2_ragbench_small",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--n-per-config", type=int, default=5)
    parser.add_argument("--run-name", type=str, default="t2-self-rag-small")
    parser.add_argument("--no-retrieval-decision", action="store_true")
    parser.add_argument("--no-retrieval-relevance", action="store_true")
    parser.add_argument("--no-evidence-support", action="store_true")
    parser.add_argument("--no-answer-usefulness", action="store_true")
    args = parser.parse_args()

    if args.prepare or not args.questions.exists():
        summary = prepare_small_subset(n_per_config=args.n_per_config)
        print("Prepared subset:", json.dumps(summary, indent=2))
        args.questions = Path(summary["questions_path"])
        args.pdf_dir = Path(summary["pdf_dir"])

    cfg = replace(CONFIG, pdf_dir=args.pdf_dir, index_dir=args.index_dir)
    assert cfg.chunk_size_tokens == 512
    assert cfg.chunk_overlap_tokens == 50
    assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.dense_top_k == 20
    assert cfg.bm25_top_k == 20
    assert cfg.final_top_k == 5
    assert cfg.hybrid_dense_weight == 0.6
    assert cfg.hybrid_bm25_weight == 0.4
    assert cfg.hybrid_method == "weighted_minmax"
    assert cfg.llm_provider == "openai_compatible"
    assert cfg.llm_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert cfg.llm_temperature == 0.0
    assert cfg.llm_num_ctx == 2048
    assert cfg.max_chars_per_chunk_in_prompt == 700
    assert not hasattr(cfg, "retrieval_decision")

    self_rag_cfg = SelfRAGConfig(
        retrieval_decision=not args.no_retrieval_decision,
        retrieval_relevance=not args.no_retrieval_relevance,
        evidence_support=not args.no_evidence_support,
        answer_usefulness=not args.no_answer_usefulness,
    )

    index_ready = (cfg.index_dir / "faiss.index").exists()
    if args.rebuild_index or not index_ready:
        print(f"Building T² indexes from {cfg.pdf_dir} -> {cfg.index_dir}")
        retriever = build_indexes(cfg, pdf_dir=cfg.pdf_dir)
    else:
        print(f"Loading T² indexes from {cfg.index_dir}")
        retriever = load_indexes(cfg)

    generator = AnswerGenerator(cfg)
    agent = SelfRAGAgent(retriever, generator, cfg, self_rag_cfg)
    logger = ExperimentLogger(args.run_name, cfg)
    questions = load_questions(args.questions)

    print(f"Logging to {logger.run_dir}")
    print("Self-RAG config:", json.dumps(self_rag_cfg.to_dict()))
    records = []
    for item in questions:
        question = agent_question_only(item)
        qid = item["question_id"]
        print(f"\n=== {item.get('dataset')} {qid}: {question}")
        record = agent.answer(question, qid)

        retrieved = []
        for raw in record.retrieved_chunks:
            chunk = retriever.chunks_by_id[raw["chunk_id"]]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    rank=raw["rank"],
                    dense_score=raw.get("dense_score"),
                    bm25_score=raw.get("bm25_score"),
                    hybrid_score=raw.get("hybrid_score") or 0.0,
                    page=chunk.page,
                    pages=list(chunk.pages),
                    document_id=chunk.document_id,
                    company=chunk.company,
                    year=chunk.year,
                    filing_type=chunk.filing_type,
                    source_filename=chunk.source_filename,
                    text=chunk.text,
                )
            )
        assert_no_gold_in_self_rag(question, item, retrieved, record)
        payload = attach_eval_metadata(record.to_dict(), item)
        path = logger.write(payload)
        records.append(payload)
        print(
            f"decision={record.retrieval_decision} relevance={record.retrieval_relevance_decision} "
            f"support={record.evidence_support_decision} useful={record.answer_usefulness_decision}"
        )
        print(
            f"retrieval_iterations={record.number_of_retrieval_iterations} "
            f"evidence_refinements={record.number_of_evidence_refinements} "
            f"llm_calls={record.number_of_llm_calls} retrieved={len(record.retrieved_chunks)}"
        )
        print(f"extra_llm_call_reasons={record.extra_llm_call_reasons}")
        if record.evidence_windows:
            print(f"evidence_windows={json.dumps(record.evidence_windows)}")
        for chunk in record.retrieved_chunks:
            print(
                f"  rank={chunk['rank']} pdf_page={chunk['page']} "
                f"doc={chunk['document_id']} hybrid={chunk['hybrid_score']:.4f}"
            )
        print(
            f"latency={record.latency_seconds}s tokens={record.total_tokens} "
            f"answer={record.answer}"
        )
        print(f"gold_answer(eval-only)={item.get('gold_answer')}")
        print(f"saved {path}")

    evaluator = Evaluator(ks=[1, 3, 5, 10])
    metrics = evaluator.evaluate_log_file(
        logger.jsonl_path,
        questions,
        output_path=logger.run_dir / "evaluation.json",
        chunks_by_id=retriever.chunks_by_id,
    )
    mean_llm = (
        sum(int(r.get("number_of_llm_calls") or 0) for r in records) / len(records) if records else 0.0
    )
    mean_ret = (
        sum(int(r.get("number_of_retrieval_iterations") or 0) for r in records) / len(records)
        if records
        else 0.0
    )
    mean_ref = (
        sum(int(r.get("number_of_evidence_refinements") or 0) for r in records) / len(records)
        if records
        else 0.0
    )
    extra = {
        "mean_llm_calls": mean_llm,
        "mean_retrieval_iterations": mean_ret,
        "mean_evidence_refinements": mean_ref,
        "self_rag_config": self_rag_cfg.to_dict(),
    }
    print("\n=== Self-RAG evaluation (gold not shown to the LLM) ===")
    print(json.dumps(metrics["retrieval"], indent=2))
    correctness = metrics["answer_correctness"]
    if isinstance(correctness, dict):
        print(
            "answer_correctness",
            json.dumps(
                {
                    "accuracy": correctness.get("accuracy"),
                    "n_correct": correctness.get("n_correct"),
                    "n": correctness.get("n"),
                    "gold_field": correctness.get("gold_field"),
                },
                indent=2,
            ),
        )
    print("by_dataset", json.dumps(metrics.get("by_dataset"), indent=2))
    print(json.dumps({"latency": metrics["latency"], "tokens": metrics["tokens"]}, indent=2))
    print("self_rag", json.dumps(extra, indent=2))
    (logger.run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "n_questions": len(records),
                "log_dir": str(logger.run_dir),
                "evaluation": metrics["retrieval"],
                "answer_correctness": correctness
                if not isinstance(correctness, dict)
                else {
                    "accuracy": correctness.get("accuracy"),
                    "n_correct": correctness.get("n_correct"),
                    "n": correctness.get("n"),
                    "gold_field": correctness.get("gold_field"),
                    "per_question": correctness.get("per_question"),
                },
                "by_dataset": metrics.get("by_dataset"),
                **extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
