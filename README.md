# Agentic RAG for financial filing QA

Local research stack for T²-RAGBench (FinQA, ConvFinQA, TAT-DQA). The **baseline** is Lumer-style **Agentic RAG**: an LLM formulates a retrieval query, then frozen hybrid search (BGE + BM25) and an evidence-only Qwen generator. **Self-RAG V2** is a separate wrapper (reflection + in-document 700-character windows) on the same retriever and generator.

Full component diagram and comparison table: [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md).

## Architecture (current baseline)

```
Original question
    ↓
LLM QueryFormulator          ← call 1: search query only; does not answer
    ↓
Deterministic metadata filters (from the original question)
    ↓
Hybrid retrieval (FAISS + BM25 → top-5)
    ↓
700-character evidence prefix per chunk
    ↓
Qwen answer generator        ← call 2: ORIGINAL question + evidence
    ↓
JSON log; gold attached after generation
```

A normal baseline question uses **two** LLM calls. Gold answers and gold evidence are never passed into formulation, retrieval, or generation.

Self-RAG is **not** part of the baseline. It currently still searches with the original question (`formulate_query()` whitespace helper). Wiring it to the same `QueryFormulator` is required before a fair 120-question SYSTEM A vs SYSTEM B comparison.

## Frozen knobs (`config.py`)

Do not change these between baseline and Self-RAG runs.

| Parameter | Value |
| --- | --- |
| `chunk_size_tokens` / `chunk_overlap_tokens` | 512 / 50 (`cl100k_base`) |
| `embedding_model` | `BAAI/bge-small-en-v1.5` |
| `dense_top_k` / `bm25_top_k` / `final_top_k` | 20 / 20 / 5 |
| Hybrid | 0.6 dense + 0.4 BM25, `weighted_minmax` |
| `max_chars_per_chunk_in_prompt` | 700 |
| `llm_model` | `qwen3:4b-instruct-2507-q4_K_M` |
| `llm_provider` / `llm_base_url` | `openai_compatible` / `http://localhost:11434/v1` |
| `llm_temperature` / `llm_num_ctx` / `llm_think` | 0.0 / 2048 / false |
| Query formulation max tokens | 64 (`agent/query_formulation.py`) |
| `max_retrieval_calls` | 2 (empty-filter retry only) |

Self-RAG flags live in `self_rag/config.py`, not in `config.CONFIG`.

## Layout

```
agent/                 BaselineAgent + QueryFormulator
retrieval/             embeddings, FAISS, BM25, hybrid fusion
ingestion/             PDF parse + page-local chunking
generation/            Ollama client + evidence-only answer prompt
self_rag/              reflection controller + 700-char window tracker
evaluation/            Recall@K, MRR, answer correctness (post-generation gold)
experiments/
  run_t2_ragbench.py           Lumer baseline runner
  run_t2_self_rag.py           Self-RAG runner
  final_120_questions.json     frozen 120-Q test set (40/40/40)
data/t2_ragbench/      development 15-Q, PDFs, sidecars
config.py
SYSTEM_ARCHITECTURE.md
```

## Ingestion and retrieval

- PDFs: **pdfplumber**. Chunks stay on one page so `page` is well-defined.
- Metadata (`company`, `year`, `filing_type`, `document_id`) from sidecar JSON or `{company}_{year}_{filing_type}.pdf`.
- Dense: BGE-small in a local FAISS inner-product index (L2-normalized).
- Sparse: BM25 on the same chunks.
- Hybrid: min-max each channel, then `0.6 * dense + 0.4 * BM25`. Logs `dense_score`, `bm25_score`, `hybrid_score`, `rank`.
- Filters: optional `company` / `year` / `filing_type` / `document_id` extracted **from the original question**. If filters return nothing, one retry without filters.

## Query formulation

`agent/query_formulation.py` is the Lumer agentic step. The LLM is told to identify entities, years, and financial concepts, preserve numbers/time, **not answer**, and output only a retrieval query. Empty model output falls back to the whitespace-normalized question. The answer prompt still uses the **original** question.

## Answer generation

Qwen sees the original question, truncated retrieved text, and source/page ids. It must use only that evidence, compute from tables when needed, and say the evidence is insufficient only when the values are missing. Default backend: local Ollama (dummy key `ollama`). Gemini/Hugging Face clients remain in `generation/llm_client.py` but are not the frozen experimental generator.

## Gold isolation and evaluation

Runners call `agent.answer(question, question_id)` with the question string only. `attach_eval_metadata` adds `gold_answer` / dataset fields to the **log** after generation.

`evaluation/evaluator.py`:

- Recall@K and MRR on `chunk.document_id == gold.document_id`
- Answer correctness via `evaluation/answer_metrics.py` (`program_answer` / `original_answer`)
- Latency and token aggregates

## Logging

Each run uses a **new** timestamped directory `logs/<run_name>_<UTC>/` (never overwrite older 15-Q or 120-Q folders):

- `{question_id}.json`
- `records.jsonl`
- `evaluation.json`
- `run_summary.json`

Baseline records include original question, formulated query, formulation prompt/raw output/latency/tokens, filters, chunk ids/ranks/scores, answer, `number_of_llm_calls` (2), and generation latency/tokens.

## Question sets

| Set | Path | Role |
| --- | --- | --- |
| Development 15-Q | `data/t2_ragbench/questions/t2_small_15.json` | Plumbing / sanity (5 FinQA + 5 ConvFinQA + 5 TAT-DQA) |
| Frozen 120-Q | `experiments/final_120_questions.json` | Final comparison set; 15-Q ids excluded; do not resample |

Hugging Face source: [`G4KMU/t2-ragbench`](https://huggingface.co/datasets/G4KMU/t2-ragbench). The adapter downloads the page PDFs needed for a run and indexes them with the frozen pipeline.

## Setup

```bash
python3 -m pip install -r requirements.txt
# Start Ollama with qwen3:4b-instruct-2507-q4_K_M on localhost:11434
python3 -m pytest tests -q
```

The first embedding run downloads `BAAI/bge-small-en-v1.5` from Hugging Face.

### Development 15-Q (existing small index)

```bash
python3 experiments/run_t2_ragbench.py \
  --questions data/t2_ragbench/questions/t2_small_15.json
```

### Frozen 120-Q Agentic baseline (do not rebuild if the index exists)

```bash
python3 experiments/run_t2_ragbench.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-ragbench-final-120-agentic
```

`--pdf-dir` / `--index-dir` / `--run-name` are existing runner flags so the 120-Q corpus is used and old conventional-RAG logs are not overwritten. Do **not** pass `--rebuild-index` when `data/indexes/t2_ragbench_final120` is already built.

Self-RAG (same questions/index when that experiment is scheduled):

```bash
python3 experiments/run_t2_self_rag.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-self-rag-final-120
```

Tiny checked-in sample (Northstar / CedarTech) is still available via `experiments/build_sample_corpus.py` and `experiments/run_baseline.py`.

## Results (do not mix conditions)

**Development 15-Q, Lumer baseline with query formulation** (`logs/t2-ragbench-small-queryform_20260823T045242Z/`):

| Metric | Value |
| --- | --- |
| Accuracy | 8/15 = 53.3% |
| Recall@1 / @3 / @5 / @10 | 1.0 |
| MRR | 1.0 |
| Mean LLM calls | 2.0 |
| Rank-1 gold document | 15/15 |

Retrieval matched the earlier 15-Q conventional baseline (query = question) at Recall@1 = 1.0 / MRR = 1.0. ConvFinQA remains 0/5 on this small set (insufficient-evidence answers despite gold-document hits).

**Do not** treat `logs/t2-ragbench-final-120_20260822T120619Z/` (43/120 = 35.8%, query = question, 1 LLM call) as the Lumer Agentic RAG baseline.

**Self-RAG V2 15-Q** (window refine; **original-question** retrieval, before formulation was added): 13/15 accuracy, Recall@1 = 1.0. That number is not comparable to SYSTEM A until Self-RAG uses `QueryFormulator`.

The frozen 120-Q **Agentic** baseline uses `experiments/final_120_questions.json` and logs under `logs/t2-ragbench-final-120-agentic_*`.

## Tests

```bash
python3 -m pytest tests -q
```
