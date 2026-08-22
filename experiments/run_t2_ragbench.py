#!/usr/bin/env python3
"""Run the frozen baseline Agentic RAG on a T²-RAGBench question file.

Gold answers and gold page-text are attached to logs after generation.
They are never passed to BaselineAgent / the retriever / the LLM.
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

from agent.agent import BaselineAgent
from config import CONFIG
from data_loader.t2_ragbench_loader import DOC_DIR, QUESTIONS_DIR, prepare_small_subset
from evaluation.evaluator import Evaluator
from evaluation.logging_utils import ExperimentLogger
from generation.answer_generator import AnswerGenerator, build_prompt
from retrieval.hybrid_retriever import RetrievedChunk
from retrieval.index_pipeline import build_indexes, load_indexes


def load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "questions" in payload:
        return payload["questions"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognized questions file format: {path}")


def agent_question_only(item: dict) -> str:
    return item["question"]


def attach_eval_metadata(record: dict, item: dict) -> dict:
    """Add dataset gold fields to the *log* after the LLM has already run."""
    out = dict(record)
    out["dataset"] = item.get("dataset")
    out["task_type"] = item.get("dataset")
    out["gold_answer"] = item.get("gold_answer")
    out["gold_evidence_reference"] = item.get("gold_evidence_reference")
    return out


def assert_no_gold_in_prompt(question: str, item: dict, retrieved_chunks) -> None:
    prompt = build_prompt(question, retrieved_chunks, CONFIG)
    gold_answer = (item.get("gold_answer") or "").strip()
    evidence = ((item.get("gold") or {}).get("evidence") or "").strip()
    lowered = prompt.lower()
    if gold_answer and len(gold_answer) >= 8 and gold_answer.lower() in lowered:
        # Short numeric golds like "13.2" can appear naturally in retrieved text.
        if not any(ch.isalpha() for ch in gold_answer):
            return
        raise AssertionError("Gold answer leaked into the LLM prompt")
    if evidence and len(evidence) > 80:
        marker = evidence[:80]
        # Dataset context may overlap the PDF page we retrieved; that is the
        # document, not a gold field injected by the runner.
        if "gold_answer" in lowered or "program_answer" in lowered:
            raise AssertionError("Gold field name leaked into the LLM prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline RAG on T²-RAGBench")
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
    parser.add_argument("--prepare", action="store_true", help="Download the 15-PDF subset first")
    parser.add_argument("--n-per-config", type=int, default=5)
    parser.add_argument("--run-name", type=str, default="t2-ragbench-small")
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
    assert cfg.llm_model == CONFIG.llm_model
    assert cfg.llm_temperature == 0.0

    index_ready = (cfg.index_dir / "faiss.index").exists()
    if args.rebuild_index or not index_ready:
        print(f"Building T² indexes from {cfg.pdf_dir} -> {cfg.index_dir}")
        retriever = build_indexes(cfg, pdf_dir=cfg.pdf_dir)
    else:
        print(f"Loading T² indexes from {cfg.index_dir}")
        retriever = load_indexes(cfg)

    generator = AnswerGenerator(cfg)
    agent = BaselineAgent(retriever, generator, cfg)
    logger = ExperimentLogger(args.run_name, cfg)
    questions = load_questions(args.questions)

    print(f"Logging to {logger.run_dir}")
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
        assert_no_gold_in_prompt(question, item, retrieved)
        payload = attach_eval_metadata(record.to_dict(), item)
        path = logger.write(payload)
        records.append(payload)
        print(f"filters={record.metadata_filters}")
        print(f"retrieved={len(record.retrieved_chunks)} chunks")
        for chunk in record.retrieved_chunks:
            print(
                f"  rank={chunk['rank']} pdf_page={chunk['page']} "
                f"doc={chunk['document_id']} hybrid={chunk['hybrid_score']:.4f} "
                f"dense={chunk['dense_score']} bm25={chunk['bm25_score']}"
            )
        print(f"latency={record.latency_seconds}s tokens={record.total_tokens}")
        print(f"answer={record.answer}")
        print(f"gold_answer(eval-only)={item.get('gold_answer')}")
        print(f"saved {path}")

    evaluator = Evaluator(ks=[1, 3, 5, 10])
    metrics = evaluator.evaluate_log_file(
        logger.jsonl_path,
        questions,
        output_path=logger.run_dir / "evaluation.json",
        chunks_by_id=retriever.chunks_by_id,
    )
    print("\n=== T²-RAGBench evaluation (gold not shown to the LLM) ===")
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
    (logger.run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "n_questions": len(records),
                "log_dir": str(logger.run_dir),
                "evaluation": metrics["retrieval"],
                "answer_correctness": metrics["answer_correctness"]
                if not isinstance(metrics["answer_correctness"], dict)
                else {
                    "accuracy": metrics["answer_correctness"].get("accuracy"),
                    "n_correct": metrics["answer_correctness"].get("n_correct"),
                    "n": metrics["answer_correctness"].get("n"),
                    "gold_field": metrics["answer_correctness"].get("gold_field"),
                },
                "by_dataset": metrics.get("by_dataset"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
