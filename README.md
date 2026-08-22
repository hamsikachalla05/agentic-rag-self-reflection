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
| `final_top_k` / `max_evidence_chunks` | 5 | context size; keep frozen |
| `hybrid_dense_weight` | 0.6 | |
| `hybrid_bm25_weight` | 0.4 | min-max normalized weighted sum |
| `hybrid_method` | `weighted_minmax` | set `rrf` for reciprocal rank fusion |
| `llm_provider` | `gemini` | experimental baseline generator |
| `llm_model` | `gemini-2.5-flash` | Gemini API; local `google/flan-t5-base` remains optional |
| `llm_temperature` | 0.0 | as close to deterministic as the API allows |
| `max_retrieval_calls` | 2 | empty-filter retry only; no reflection |

Do not change embedding model, chunking, retriever, or context size between baseline and Self-RAG runs. Changing the LLM is a separate experimental condition and must be documented.

## Ingestion and chunking

- PDFs are parsed with **pdfplumber** (text + tables). Page numbers are retained.
- Metadata (`company`, `year`, `filing_type`, `document_id`) comes from a sidecar JSON or from the filename pattern `{company}_{year}_{filing_type}.pdf`. Missing fields stay missing.
- Chunks are ~512 tokens with 50-token overlap, **split within a page** so `page` stays exact for evaluation. Each chunk has a stable `chunk_id` plus `document_id`, `page` / `pages`, `company`, `year`, `source_filename`, and `text`.

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

Default experimental generator: **Gemini 2.5 Flash** (`llm_provider = "gemini"`, `llm_model = "gemini-2.5-flash"`). The API key is read from `GEMINI_API_KEY` and is never stored in the repository.

The previous local FLAN-T5 backend is still available:

```python
# config.py
llm_provider = "huggingface_local"
llm_model = "google/flan-t5-base"
```

An OpenAI-compatible endpoint can use `llm_provider = "openai_compatible"` with `LLM_BASE_URL` / `LLM_API_KEY`. Evidence passed into the prompt is truncated per chunk (`max_chars_per_chunk_in_prompt`) so every backend sees the same retrieved text.

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

The first embedding run downloads `BAAI/bge-small-en-v1.5` from Hugging Face. Gemini generation requires an API key (do not commit the key):

**Windows Git Bash / macOS / Linux**

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

**PowerShell**

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

Local FLAN-T5 (`llm_provider = "huggingface_local"`) additionally downloads `google/flan-t5-base`.

## Prototype end-to-end run

Checked-in sample PDFs (Northstar 2023, CedarTech 2022) and 10 questions. Gold pages are used **only** by `evaluation/evaluator.py`. JSON records from the CPU prototype are in `logs/examples/`.

| Metric | Value |
| --- | --- |
| Recall@1 | 0.80 |
| Recall@3 | 1.00 |
| MRR | 0.88 |
| Mean latency | 0.40 s |
| Mean total tokens | 520 |

Rank-1 retrieved page matched the gold page for 8/10 questions; the other two matched within the top 3. Generated answers copied the correct figure or name for most questions (revenue, employees, office, operating income, segment, dividend, CEO, acquisition price). FLAN-T5-base still misses some numeric details (effective tax rate, one percentage). That is expected for this small frozen generator: Self-RAG should wrap this retriever rather than replace it.

Example (`q01`): filters `{company, year}` from the question; rank-1 chunk is page 3 with dense/BM25/hybrid scores; answer `$12.40 billion`.

## Prototype vs full corpus

The checked-in sample PDFs and 10 questions are an end-to-end prototype so you can confirm that retrieved chunks map to the correct source pages. The same code paths scale to a full filing corpus: parse directory → chunk → embed → BM25 → hybrid retrieve → agent → log.

## Adding Self-RAG later

Keep this baseline frozen. Add reflection as a new package (for example `reflection/`) that may:

- grade retrieved documents
- critique a draft answer
- optionally issue a **new** retrieval call through the **same** `HybridRetriever`

Do not change `config.py` defaults for chunk size, overlap, embedding model, hybrid weights, or `final_top_k` when comparing against this retrieval baseline.

## T²-RAGBench (dataset adapter)

Hugging Face dataset: [`G4KMU/t2-ragbench`](https://huggingface.co/datasets/G4KMU/t2-ragbench) (`FinQA`, `ConvFinQA`, `TAT-DQA`).

The Arrow tables contain questions, answers, and page text. **Page PDFs are files in the same Hugging Face repo** (`data/FinQA/{split}/pdf/...`, `data/ConvFinQA/pdf/...`, `data/TAT-DQA/{split}/raw/...`), not binary columns. The adapter downloads only the PDFs needed for a run and indexes them with the **frozen** baseline pipeline.

Gold `program_answer` / `context` are stored on the question record for evaluation and are attached to logs **after** generation. They are not passed to retrieval, embeddings, BM25, the agent, or the LLM.

Evidence mapping: each PDF is one source page keyed by `context_id` (= our `document_id`). Recall/MRR use `chunk.document_id == gold.document_id`. Dataset `page_number` is the 10-K page label; pdfplumber’s page index inside the single-page PDF is 1. Token overlap vs `gold.evidence` is a secondary check because 512-token chunks are not identical to the dataset `context` string.

```bash
python3 experiments/run_t2_ragbench.py --prepare --rebuild-index
```

This prepares 5 FinQA test + 5 ConvFinQA turn_0 + 5 TAT-DQA test questions (15 unique page PDFs), then runs the frozen retrieval baseline with the configured generator (default: Gemini 2.5 Flash).

Do not process the full benchmark until this 15-question path is verified.

## Tests

```bash
python3 -m pytest tests -q
```
