#!/usr/bin/env python3
"""Run the baseline Agentic RAG pipeline on a question file.

Ground-truth fields in the question file are ignored by the agent and used
only by the evaluation module after answers are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import BaselineAgent
from config import CONFIG
from evaluation.evaluator import Evaluator
from evaluation.logging_utils import ExperimentLogger
from generation.answer_generator import AnswerGenerator
from retrieval.index_pipeline import build_indexes, load_indexes


def load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "questions" in payload:
        return payload["questions"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognized questions file format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline Agentic RAG")
    parser.add_argument("--questions", type=Path, default=CONFIG.questions_path)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--pdf-dir", type=Path, default=CONFIG.pdf_dir)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="baseline")
    args = parser.parse_args()

    index_ready = (CONFIG.index_dir / "faiss.index").exists()
    if args.rebuild_index or not index_ready:
        print(f"Building indexes from {args.pdf_dir} ...")
        retriever = build_indexes(CONFIG, pdf_dir=args.pdf_dir)
    else:
        print(f"Loading indexes from {CONFIG.index_dir} ...")
        retriever = load_indexes(CONFIG)

    generator = AnswerGenerator(CONFIG)
    agent = BaselineAgent(retriever, generator, CONFIG)
    logger = ExperimentLogger(args.run_name, CONFIG)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    print(f"Logging to {logger.run_dir}")
    records = []
    for item in questions:
        qid = item["question_id"]
        question = item["question"]
        print(f"\n=== {qid}: {question}")
        record = agent.answer(question, qid)
        path = logger.write(record.to_dict())
        records.append(record.to_dict())
        print(f"filters={record.metadata_filters}")
        print(f"retrieved={len(record.retrieved_chunks)} chunks")
        for chunk in record.retrieved_chunks:
            print(
                f"  rank={chunk['rank']} page={chunk['page']} "
                f"doc={chunk['document_id']} hybrid={chunk['hybrid_score']:.4f} "
                f"dense={chunk['dense_score']} bm25={chunk['bm25_score']}"
            )
        print(f"latency={record.latency_seconds}s tokens={record.total_tokens}")
        print(f"answer={record.answer}")
        print(f"saved {path}")

    evaluator = Evaluator()
    metrics = evaluator.evaluate_log_file(
        logger.jsonl_path,
        questions,
        output_path=logger.run_dir / "evaluation.json",
    )
    print("\n=== Retrieval evaluation (gold pages; not shown to the LLM) ===")
    print(json.dumps(metrics["retrieval"], indent=2))
    print(json.dumps({"latency": metrics["latency"], "tokens": metrics["tokens"]}, indent=2))
    (logger.run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "n_questions": len(records),
                "log_dir": str(logger.run_dir),
                "evaluation": metrics["retrieval"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
