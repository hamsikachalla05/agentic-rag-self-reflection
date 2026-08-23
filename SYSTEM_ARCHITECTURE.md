# System architecture

Research stack for financial QA on T²-RAGBench. The implementation follows Lumer et al. (vector-based **Agentic RAG**: LLM search-query formulation, then hybrid retrieval) and layers optional **Self-RAG-inspired reflection** on the same retriever and generator.

This document describes the **current code**, not a target design. Where intended future wiring differs from what is running now, that is stated explicitly.

---

## 1. Systems being compared

| | **SYSTEM A — Lumer-style Agentic RAG (baseline)** | **SYSTEM B — Self-RAG V2** |
| --- | --- | --- |
| Query | LLM `QueryFormulator` before search | **Today:** original question (whitespace helper). **Intended:** same `QueryFormulator` |
| Filters | Deterministic regex / string match on the **original question** | Same helper (`extract_metadata_filters`) |
| Retrieve | Frozen hybrid (FAISS + BM25) | Same `HybridRetriever`, one hybrid pass |
| Evidence | Top-5, first 700 characters per chunk | Same initial window; if UNSUPPORTED, slide a later 700-char slice in the same document / chunker sibling |
| Answer | Evidence-only Qwen prompt | Same generator; optional revision if NOT_USEFUL |
| LLM calls / typical question | **2** (formulate + answer) | **~5–8** (formulate not yet wired; reflection + answer + optional refine) |

Shared and frozen for a fair comparison:

- `experiments/final_120_questions.json` (40 FinQA / 40 ConvFinQA / 40 TAT-DQA)
- same PDFs and indexes (`data/t2_ragbench/documents_final120`, `data/indexes/t2_ragbench_final120`)
- BGE `BAAI/bge-small-en-v1.5`, chunk 512 / overlap 50, hybrid 0.6 / 0.4 `weighted_minmax`, `final_top_k=5`, 700-char window
- Qwen `qwen3:4b-instruct-2507-q4_K_M` via Ollama `http://localhost:11434/v1`, `temperature=0`, `num_ctx=2048`, thinking off
- same evaluator and gold isolation

---

## 2. End-to-end pipeline

```
                    ┌─────────────────────────────────────────┐
                    │  T²-RAGBench questions (string only)    │
                    │  gold attached AFTER generation         │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │           BaselineAgent.answer          │
                    │                                         │
                    │  1. QueryFormulator (LLM, max 64 tok)   │
                    │     question → retrieval query          │
                    │  2. extract_metadata_filters(question)  │
                    │     year / company / filing / doc_id    │
                    │  3. HybridRetriever.retrieve(query)     │
                    │     dense@20 + BM25@20 → hybrid top-5   │
                    │     optional empty-filter retry         │
                    │  4. AnswerGenerator.generate            │
                    │     ORIGINAL question + evidence        │
                    └────────────────────┬────────────────────┘
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              │  Self-RAG V2 (optional wrapper; not in baseline)     │
              │  Always hybrid-retrieve first (preserve rank-1)      │
              │  Reflect: NEED_RETRIEVAL / RELEVANT / SUPPORTED /    │
              │           USEFUL  (single-token labels)              │
              │  If UNSUPPORTED: slide 700-char window in-document   │
              │  Do not re-query FAISS/BM25 with a new search query  │
              └──────────────────────────┬──────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  ExperimentLogger  →  Evaluator         │
                    │  records.jsonl / evaluation.json        │
                    │  Recall@K, MRR, numeric answer match    │
                    └─────────────────────────────────────────┘
```

---

## 3. Component map

| Layer | Path | Role |
| --- | --- | --- |
| Config | `config.py` | Frozen chunk / retrieve / generate knobs. **No** Self-RAG flags. |
| Ingest | `ingestion/pdf_parser.py`, `ingestion/chunker.py` | pdfplumber; page-local ~512-token chunks, 50 overlap. |
| Index | `retrieval/index_pipeline.py` | Build/load FAISS + BM25 + `chunks.jsonl`. |
| Dense | `retrieval/embeddings.py`, `retrieval/vector_store.py` | BGE-small, L2-normalized inner product. |
| Sparse | `retrieval/bm25.py` | `rank_bm25` on the same chunks. |
| Hybrid | `retrieval/hybrid_retriever.py` | Filter → dense@20 + BM25@20 → min-max weighted fusion → top-5. |
| Query formulation | `agent/query_formulation.py` | Lumer agentic step: LLM retrieval query, not an answer. |
| Baseline controller | `agent/agent.py` | `BaselineAgent.answer`. |
| Generation | `generation/answer_generator.py` | Evidence-only prompt; 700-char prefix; original question. |
| LLM I/O | `generation/llm_client.py` | Ollama OpenAI-compatible client. |
| Self-RAG config | `self_rag/config.py` | Reflection flags and caps (`max_llm_calls=8`, …). |
| Self-RAG controller | `self_rag/controller.py` | Wrap retrieve + reflect + window refine + generate. |
| Reflection | `self_rag/reflector.py`, `self_rag/prompts.py` | Four label classifiers. |
| Evidence windows | `self_rag/windows.py` | Later 700-char slice or same-document chunker sibling. |
| Eval | `evaluation/evaluator.py`, `evaluation/answer_metrics.py` | Recall@K / MRR on gold `document_id`; numeric/string correctness. |
| Logs | `evaluation/logging_utils.py` | Timestamped dir: one JSON per question + `records.jsonl`. |
| Runners | `experiments/run_t2_ragbench.py`, `experiments/run_t2_self_rag.py` | Baseline vs Self-RAG on a question file. |
| Manifest | `experiments/final_120_questions.json` | Frozen 120-Q final set (dev 15-Q excluded). |

---

## 4. SYSTEM A — Lumer-style Agentic RAG (baseline)

### 4.1 Query formulation (the agentic retrieval action)

**File:** `agent/query_formulation.py`  
**When:** first LLM call in `BaselineAgent.answer`, **before** `HybridRetriever.retrieve`.

- **Input:** original question string only.
- **Output:** one-line retrieval query for dense + BM25.
- **Must not:** answer the question, invent entities/years/figures, see gold.
- **Parse:** first non-empty line; if empty, fall back to whitespace-normalized original question.
- **Budget:** `QUERY_FORMULATION_MAX_TOKENS = 64`. Same Qwen, temperature 0.

Answer generation still receives the **original question**, not the formulated query.

### 4.2 Metadata filters (non-LLM)

**File:** `agent/agent.py` → `extract_metadata_filters(question)`

Applied to the **original question**, not the formulated query:

- single explicit year (`19xx` / `20xx`)
- known company name from the index
- filing type (`10-K`, `10-Q`, `20-F`, “annual report”)
- document id substring if present

No invented filters. If filters return zero chunks and `retry_without_filters_if_empty`, one retry with the **same formulated query** and `filters=None`. That is not LLM-driven iteration.

### 4.3 Hybrid retrieval (frozen)

**File:** `retrieval/hybrid_retriever.py`

1. Restrict candidate ids by filters (if any).
2. Dense search `dense_top_k=20`.
3. BM25 search `bm25_top_k=20`.
4. Min-max normalize each channel; `hybrid = 0.6 * dense + 0.4 * BM25`.
5. Return `final_top_k=5` with `rank`, `dense_score`, `bm25_score`, `hybrid_score`, `document_id`, `page`.

Query formulation changes **only the query string** passed into this function. Fusion, k, embeddings, and indexes stay fixed.

### 4.4 Answer generation

**File:** `generation/answer_generator.py`

- Pack top `max_evidence_chunks=5` chunks.
- Truncate each chunk from the **start** to 700 characters (`max_chars_per_chunk_in_prompt`).
- Prompt: system “use only retrieved evidence” + `Question:` + `Evidence:` + `Answer:`.
- Second LLM call. `llm_calls=1` at the generator; baseline record totals **2**.

### 4.5 Baseline LLM call budget

| Call | Purpose |
| --- | --- |
| 1 | Formulate retrieval query |
| 2 | Generate final answer from evidence |

HTTP retries are not extra reasoning steps.

---

## 5. SYSTEM B — Self-RAG V2 (wrapper)

**Does not replace** hybrid search. It wraps the same retriever and `AnswerGenerator`.

### 5.1 Current vs intended query path

| | Current code | Intended fair comparison |
| --- | --- | --- |
| Query used for hybrid search | `BaselineAgent.formulate_query()` = whitespace on the original question | Same `QueryFormulator` as SYSTEM A |
| Why | Self-RAG files were frozen while formulation was added to the baseline | SYSTEM B = A + reflection + window refine |

`QueryFormulator` is exported from `agent/` so Self-RAG can call `formulate_retrieval_query()` later **without** changing FAISS/BM25.

### 5.2 Control loop (current)

1. Always retrieve once with the frozen hybrid retriever (empty-filter retry only). **Original ranks are stored for Recall@K.**
2. Build a 700-char evidence view (`EvidenceWindowTracker`).
3. **Retrieval decision:** `NEED_RETRIEVAL` / `NO_RETRIEVAL` (does not drop rank-1).
4. **Relevance:** `RELEVANT` / `IRRELEVANT`.
5. **Evidence support:** `SUPPORTED` / `UNSUPPORTED`.
   - `SUPPORTED` → keep current window.
   - `UNSUPPORTED` (or IRRELEVANT) → slide a later 700-char window in the **same retrieved chunk**, or a same-document **chunker sibling**. Not a new hybrid query. Cap: `max_evidence_refinements=2`.
6. Generate answer on the current view (original question + windowed text).
7. **Usefulness:** `USEFUL` / `NOT_USEFUL`. Optional revision / extra window under `max_answer_attempts=2` and `max_llm_calls=8`.

Caps live in `self_rag/config.py`, not in `config.CONFIG`.

---

## 6. Data, gold isolation, evaluation

### 6.1 Question sets

| Set | Path | Use |
| --- | --- | --- |
| Development 15-Q | `data/t2_ragbench/questions/t2_small_15.json` | Sanity / plumbing (5+5+5). **Not** the final comparison set. |
| Final 120-Q | `experiments/final_120_questions.json` | Frozen test: 40/40/40, seed 20260822, 15-Q ids excluded. |

Runners pass **only** `item["question"]` into `agent.answer(question, question_id)`. Gold is copied onto the log in `attach_eval_metadata` **after** generation.

### 6.2 What the models never see

Gold field names and gold answers are not in:

- query-formulation prompt
- hybrid query
- answer prompt
- Self-RAG reflection prompts

Numeric golds may appear inside retrieved PDF text; that is document content, not injected labels.

### 6.3 Metrics

- **Retrieval:** hit if a retrieved chunk’s `document_id` is in `gold.document_ids`. Recall@1/3/5/10, MRR. Self-RAG scores the **original** retrieval list, not window slides.
- **Answers:** `evaluation/answer_metrics.py` against `program_answer` / `original_answer` (numbers, percents, yes/no, list forms).
- **Cost:** wall latency, token totals, `number_of_llm_calls`.

### 6.4 Logs (never overwrite older runs)

Each run gets `logs/{run_name}_{UTC}/`:

- `{question_id}.json`
- `records.jsonl`
- `evaluation.json`
- `run_summary.json`

Baseline formulation logs include `retrieval_query`, `query_formulation_prompt`, `query_formulation_raw_output`, latencies/tokens for formulation vs generation, filters, chunk ids/ranks/scores, answer, `number_of_llm_calls`.

---

## 7. Frozen experimental knobs

| Knob | Value |
| --- | --- |
| Chunk size / overlap | 512 / 50 tokens, `cl100k_base` |
| Embedding | `BAAI/bge-small-en-v1.5` |
| Dense / BM25 / final k | 20 / 20 / 5 |
| Hybrid | 0.6 dense + 0.4 BM25, `weighted_minmax` |
| Evidence window | 700 characters from start of each packed chunk |
| Generator | `qwen3:4b-instruct-2507-q4_K_M`, T=0, `num_ctx=2048`, think off |
| Formulation max tokens | 64 |
| Answer max new tokens | 512 (`config.llm_max_new_tokens`) |
| Empty-filter retry | `max_retrieval_calls=2` |

Changing any of these is a new experimental condition, not a silent patch.

---

## 8. What is *not* in the architecture

- No LLM query expansion / multi-hop search in the baseline.
- No LLM-chosen chunk set (ranking is hybrid scores).
- No Self-RAG in SYSTEM A (no retrieval grading, no usefulness loop, no window slides).
- No gold-in-the-loop retrieval or question selection.
- Self-RAG does **not** replace FAISS/BM25 with a new retriever.

---

## 9. How to run (existing runners)

Development 15-Q baseline (existing small index):

```bash
python experiments/run_t2_ragbench.py \
  --questions data/t2_ragbench/questions/t2_small_15.json
```

Final 120-Q baseline (frozen corpus/index; extra flags are existing argparse, not a behavior change):

```bash
python experiments/run_t2_ragbench.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-ragbench-final-120-agentic
```

Do **not** pass `--rebuild-index` when the frozen `t2_ragbench_final120` index already exists.

Self-RAG: `experiments/run_t2_self_rag.py` with the same questions/index when that comparison is scheduled.

---

## 10. Implementation status (as of this document)

| Item | Status |
| --- | --- |
| LLM query formulation on baseline | Implemented and used in `BaselineAgent.answer` |
| 15-Q development sanity (formulation on) | Recall@1 = 1.0, MRR = 1.0, accuracy 8/15 |
| Final 120-Q **Agentic** baseline | Runner + frozen index; results live in a new `logs/t2-ragbench-final-120-agentic_*` directory when that job finishes |
| Old 120-Q **conventional** RAG (query = question) | `logs/t2-ragbench-final-120_20260822T120619Z/` — **not** SYSTEM A |
| Self-RAG V2 on original-question retrieval | Implemented; 120-Q was run **before** formulation was added to the baseline |
| Self-RAG using `QueryFormulator` | **Not wired yet** — required before SYSTEM A vs B on the same 120 questions |
