# Baseline Agentic RAG for financial annual-report QA

Research baseline for long-document financial QA. This repository implements **only the baseline**: hybrid retrieval plus a lightweight controller and an evidence-grounded LLM. **Self-RAG, retrieval grading, answer critique, and experimental query rewriting are not included** and should be added later on top of this pipeline without changing chunking, embeddings, the retriever, context size, or the LLM.

## Architecture

```
Question
    ↓
Agent (query = question; metadata filters only if explicit)
    ↓
Metadata filtering (optional; never invented)
    ↓
Hybrid retrieval
    ├── Dense semantic vector search (FAISS + BGE)
    └── BM25 keyword search
    ↓
Weighted score combination / ranking
    ↓
Top-k evidence
    ↓
LLM (evidence-only prompt)
    ↓
Final answer + JSON experiment log
```

The retriever is a stable interface (`retrieval/hybrid_retriever.py`). A future reflection module should wrap the agent or generator; it should not fork a second retrieval stack.

## Project layout

```
project/
├── data/
│   ├── reports/                 # annual-report PDFs
│   ├── questions/               # questions (+ gold pages for eval only)
│   └── indexes/                 # built locally, not required in git
├── ingestion/
│   ├── pdf_parser.py
│   └── chunker.py
├── retrieval/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25.py
│   ├── hybrid_retriever.py
│   └── index_pipeline.py
├── agent/
│   └── agent.py
├── generation/
│   ├── llm_client.py
│   └── answer_generator.py
├── evaluation/
│   ├── evaluator.py
│   └── logging_utils.py
├── experiments/
│   ├── build_sample_corpus.py
│   └── run_baseline.py
├── logs/
├── config.py
├── requirements.txt
└── README.md
```

## Configuration (frozen baseline knobs)

All research-critical parameters live in [`config.py`](config.py):

| Parameter | Default | Notes |
| --- | --- | --- |
| `chunk_size_tokens` | 512 | tiktoken `cl100k_base` |
| `chunk_overlap_tokens` | 50 | |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | free, local, 512-token context |
| `dense_top_k` | 20 | |
| `bm25_top_k` | 20 | |
| `final_top_k` / `max_evidence_chunks` | 8 | context size; keep frozen |
| `hybrid_dense_weight` | 0.6 | |
| `hybrid_bm25_weight` | 0.4 | min-max normalized weighted sum |
| `hybrid_method` | `weighted_minmax` | set `rrf` for reciprocal rank fusion |
| `llm_model` | `Qwen/Qwen2.5-0.5B-Instruct` | free local Hugging Face model (long enough context for evidence) |
| `llm_temperature` | 0.0 | |
| `max_retrieval_calls` | 2 | empty-filter retry only; no reflection |

Do not change embedding model, chunking, retriever, context size, or LLM between baseline and Self-RAG runs unless the change is documented as a new experimental condition.

## Ingestion and chunking

- PDFs are parsed with **pdfplumber** (text + tables). Page numbers are retained.
- Metadata (`company`, `year`, `filing_type`, `document_id`) comes from a sidecar JSON or from the filename pattern `{company}_{year}_{filing_type}.pdf`. Missing fields stay missing.
- Chunks are ~512 tokens with 50-token overlap. Each chunk has a stable `chunk_id` plus `document_id`, `page` / `pages`, `company`, `year`, `source_filename`, and `text`.

## Retrieval

- **Dense:** `BAAI/bge-small-en-v1.5` embeddings in a local FAISS inner-product index (L2-normalized vectors).
- **BM25:** `rank_bm25` over the same chunks.
- **Hybrid:** min-max normalize each channel independently, then `0.6 * dense + 0.4 * BM25`. Chunks present in only one list get `0` on the other channel. Every returned chunk logs `dense_score`, `bm25_score`, `hybrid_score`, and `rank`.
- **Filters:** optional `company`, `year`, `filing_type`, `document_id` applied **before** scoring. If the question does not contain a cue, no filter is applied.

## Agent (baseline)

The controller is intentionally small:

1. Retrieval query = stripped user question (no rewrite module).
2. Heuristic metadata extraction from explicit years, known company names, filing-type phrases, and document IDs already in the index.
3. One hybrid retrieval call. If filters yield zero hits and `max_retrieval_calls >= 2`, retry once without filters.
4. Pass retrieved chunks (not gold context) to the generator.

## Answer generation

The LLM receives the original question, retrieved chunk text, and source/page identifiers. It is instructed to answer only from evidence, avoid unsupported claims, admit insufficient evidence, and cite pages. Dataset gold answers/contexts are **never** injected.

Default generator: local `Qwen/Qwen2.5-0.5B-Instruct` (CPU). For a stronger but still local/OpenAI-compatible endpoint, set `llm_provider = "openai_compatible"` and `LLM_BASE_URL` / `LLM_API_KEY`. Freeze that choice before the main study.

Token usage is logged as input / output / total tokens. There is no monetary cost metric.

## Logging

Every question writes a JSON record (and a JSONL file) under `logs/<run>/`:

```json
{
  "question_id": "...",
  "question": "...",
  "retrieval_query": "...",
  "metadata_filters": {},
  "retrieved_chunks": [
    {
      "chunk_id": "...",
      "rank": 1,
      "dense_score": 0.0,
      "bm25_score": 0.0,
      "hybrid_score": 0.0,
      "page": 1,
      "document_id": "..."
    }
  ],
  "answer": "...",
  "latency_seconds": 0.0,
  "input_tokens": 0,
  "output_tokens": 0,
  "total_tokens": 0,
  "number_of_retrieval_calls": 1,
  "number_of_llm_calls": 1
}
```

End-to-end latency starts before processing the question and stops after the final answer. Retrieval and LLM timings are also stored.

## Evaluation (separate module)

`evaluation/evaluator.py` does **not** sit inside the retriever. Gold document/page IDs in the question file are used only after logging:

- Recall@K and MRR when gold pages/document IDs are provided
- latency and token aggregates
- placeholders for answer correctness and evidence support

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 experiments/build_sample_corpus.py
python3 experiments/run_baseline.py --rebuild-index
```

Place real annual-report PDFs in `data/reports/` using `{Company}_{Year}_{filing-type}.pdf`, add questions under `data/questions/`, then rebuild the index.

First run downloads `BAAI/bge-small-en-v1.5` and `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face.

## Prototype vs full corpus

The checked-in sample PDFs and 10 questions are an end-to-end prototype so you can confirm that retrieved chunks map to the correct source pages. The same code paths scale to a full filing corpus: parse directory → chunk → embed → BM25 → hybrid retrieve → agent → log.

## Adding Self-RAG later

Keep this baseline frozen. Add reflection as a new package (for example `reflection/`) that may:

- grade retrieved documents
- critique a draft answer
- optionally issue a **new** retrieval call through the **same** `HybridRetriever`

Do not change `config.py` defaults for chunk size, overlap, embedding model, hybrid weights, `final_top_k`, or LLM when comparing against this baseline.

## Tests

```bash
python3 -m pytest tests -q
```
