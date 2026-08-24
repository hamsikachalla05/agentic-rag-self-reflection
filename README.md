# Reflective Agentic RAG for Financial Filing QA

A research implementation and evaluation of **Self-RAG-style reflection and adaptive evidence refinement** over a Lumer-style **Agentic RAG** baseline for financial filing question answering.

The system is evaluated on **T²-RAGBench**, using questions from **FinQA, ConvFinQA, and TAT-DQA**.

The study compares:

- **Lumer Agentic RAG** — LLM query formulation + frozen hybrid retrieval + evidence-only generation
- **Full Self-RAG V2** — the same baseline retrieval pipeline with reflection and adaptive in-document evidence refinement
- Four component ablations:
  - `NEED_RETRIEVAL OFF`
  - `RELEVANT OFF`
  - `SUPPORTED OFF`
  - `USEFUL OFF`

On the frozen 120-question evaluation set, Full Self-RAG V2 achieves **50.8% accuracy**, compared with **37.5%** for the Lumer Agentic RAG baseline, an improvement of **13.3 percentage points**.

Full component diagram and comparison table: `SYSTEM_ARCHITECTURE.md`.

---

## Research Contributions

This repository accompanies a controlled study of reflective Agentic RAG for financial question answering.

The study investigates:

1. **Whether Self-RAG-style reflection improves financial QA** over a Lumer-style Agentic RAG baseline.
2. **Whether the improvement comes from post-retrieval evidence use rather than retrieval itself**, by keeping the query formulation, hybrid retrieval system, index, and generator fixed.
3. **The contribution of individual reflection components** through four one-component ablations:
   - retrieval decision
   - retrieval relevance
   - evidence support
   - answer usefulness
4. **The accuracy–efficiency trade-off** introduced by reflective inference through measurements of latency, tokens, LLM calls, and evidence refinements.

---

## System Overview

### Lumer Agentic RAG Baseline

The baseline consists of an LLM query formulation step followed by deterministic hybrid retrieval and evidence-only answer generation.

```text
Original question
       |
       v
LLM QueryFormulator
(call 1: retrieval query only; does not answer)
       |
       v
Deterministic metadata filters
(from the original question)
       |
       v
Hybrid retrieval
(FAISS + BM25 -> top-5)
       |
       v
700-character evidence prefix
per retrieved chunk
       |
       v
Qwen answer generator
(call 2: original question + evidence)
       |
       v
JSON log
(gold attached after generation)
```

A normal baseline question uses **two LLM calls**.

Gold answers and gold evidence are never provided to query formulation, retrieval, or generation.

### Self-RAG V2

Self-RAG V2 uses the **same QueryFormulator, frozen hybrid retrieval index, and Qwen generator** as the baseline. It is a separate wrapper (reflection + in-document 700-character window slides) on the same retriever and generator.

Reflection and adaptive in-document evidence refinement are added after retrieval.

```text
Original question
       |
       v
LLM QueryFormulator
       |
       v
Frozen hybrid retrieval
       |
       v
Retrieved evidence
       |
       v
Self-RAG reflection
       |
       +--> NEED_RETRIEVAL
       |
       +--> RELEVANT
       |
       +--> SUPPORTED
       |
       +--> USEFUL
       |
       v
Adaptive in-document
700-character windows
       |
       v
Evidence-only generation
       |
       v
Final answer
```

Self-RAG is therefore **not part of the baseline**. It is a separate wrapper around the same retrieval/generation pipeline.

---

## Frozen Experimental Configuration

The following parameters are kept fixed between the baseline and Self-RAG runs (see `config.py`). Do not change these between baseline and Self-RAG runs.

| Parameter                          | Value                           |
| ----------------------------------- | -------------------------------- |
| `chunk_size_tokens`                 | 512                              |
| `chunk_overlap_tokens`              | 50                                |
| Tokenizer                           | `cl100k_base`                    |
| Embedding model                     | `BAAI/bge-small-en-v1.5`         |
| Dense top-k                         | 20                                |
| BM25 top-k                          | 20                                |
| Final top-k                         | 5                                 |
| Hybrid weighting                    | 0.6 dense + 0.4 BM25              |
| Hybrid normalization                | `weighted_minmax`                |
| Maximum characters/chunk in prompt  | 700                               |
| LLM                                 | `qwen3:4b-instruct-2507-q4_K_M`   |
| LLM provider                        | `openai_compatible`               |
| LLM backend / base URL              | Local Ollama, `http://localhost:11434/v1` |
| LLM temperature                     | 0.0                                |
| LLM context (`num_ctx`)             | 2048                               |
| Thinking (`llm_think`)              | disabled                           |
| Query formulation max tokens        | 64 (`agent/query_formulation.py`)  |
| Maximum retrieval calls             | 2 (empty-filter retry only)        |

Self-RAG-specific flags are stored in `self_rag/config.py` and are not part of `config.CONFIG` / the frozen baseline configuration.

---

## Repository Structure

```text
agent/                 BaselineAgent + QueryFormulator
retrieval/              embeddings, FAISS, BM25, hybrid fusion
ingestion/              PDF parsing, page-local chunking
generation/             Ollama client, evidence-only answer prompt
self_rag/               reflection controller, adaptive 700-char window tracker
evaluation/             Recall@K, MRR, answer correctness (post-generation gold), latency/token aggregation
experiments/
  run_t2_ragbench.py            Lumer baseline runner
  run_t2_self_rag.py            Self-RAG runner
  final_120_questions.json      frozen 120-Q test set (40/40/40)
  self_rag_ablation_correctness_matrix.csv
  build_sample_corpus.py        tiny checked-in sample corpus builder (Northstar / CedarTech)
  run_baseline.py               tiny sample corpus runner
data/t2_ragbench/       development 15-Q set, PDFs, sidecars
config.py
SYSTEM_ARCHITECTURE.md
```

---

## Ingestion and Retrieval

### Document ingestion

PDFs are parsed using `pdfplumber`. Chunks remain within a single page so that page-level source information remains well-defined.

Document metadata includes:

- `company`
- `year`
- `filing_type`
- `document_id`

Metadata is obtained from sidecar JSON files or filenames following:

```text
{company}_{year}_{filing_type}.pdf
```

### Dense retrieval

Dense retrieval uses `BAAI/bge-small-en-v1.5`. Embeddings are stored in a local FAISS inner-product index with L2 normalization.

### Sparse retrieval

BM25 is computed over the same document chunks.

### Hybrid retrieval

Dense and BM25 scores are independently min-max normalized and combined using:

```text
0.6 * dense + 0.4 * BM25
```

The final retrieval stage returns the top-5 chunks. Retrieval logs contain dense score, BM25 score, hybrid score, rank, document ID, and page information.

### Metadata filtering

Optional filters (company, year, filing type, document ID) are extracted from the **original question**. If the filters return no results, one retry is performed without the filters.

---

## Query Formulation

`agent/query_formulation.py` implements the Lumer-style Agentic RAG query formulation step.

The LLM is instructed to:

- identify relevant entities
- identify years
- identify financial concepts
- preserve numbers and time information
- formulate a retrieval-oriented query
- **not answer the question**
- output only the retrieval query

If the model returns an empty query, the whitespace-normalized original question is used as a fallback.

Importantly, the answer generator still receives the **original question**, rather than the formulated query.

---

## Answer Generation

The Qwen generator receives:

- the original question
- truncated retrieved evidence
- source/page identifiers

The generator is instructed to use only the supplied evidence, compute from tables when needed, and perform calculations from financial tables when required. If the required values are genuinely absent, it may state that the evidence is insufficient.

The frozen experimental backend is local Ollama running `qwen3:4b-instruct-2507-q4_K_M`, default endpoint `http://localhost:11434/v1` (dummy key `ollama`). Gemini/Hugging Face clients remain in `generation/llm_client.py` but were not used as the frozen experimental generator.

---

## Self-RAG Reflection

Self-RAG V2 introduces four reflection components:

### NEED_RETRIEVAL
Determines whether additional retrieval/evidence is required.

### RELEVANT
Assesses whether the available evidence is relevant to the question.

### SUPPORTED
Assesses whether the proposed reasoning/answer is supported by the available evidence.

### USEFUL
Assesses whether the available evidence is sufficient and useful for producing the answer.

When additional evidence is required, Self-RAG can refine the evidence using **adaptive in-document 700-character windows**. The four components are individually removed in the ablation experiments.

---

## Gold Isolation and Evaluation

The experiment maintains strict gold-answer isolation. Runners call `agent.answer(question, question_id)` using the question string only. `attach_eval_metadata` adds `gold_answer` / dataset fields to the log **only after generation**.

Gold answers and gold evidence are therefore not available to query formulation, retrieval, reflection, or answer generation.

### Retrieval metrics

Retrieval is evaluated using the gold document ID (`chunk.document_id == gold.document_id`). Reported retrieval metrics include Recall@1, Recall@3, Recall@5, Recall@10, and MRR.

### Answer correctness

Answer correctness is evaluated using `evaluation/evaluator.py` and `evaluation/answer_metrics.py`, supporting the dataset's `program_answer` and `original_answer` representations. Latency and token usage are aggregated from the individual run records.

### Evaluator audit

The original scorer accepted a prediction when any parsed numerical quantity in the free-form response matched the gold answer under the evaluator's numerical equivalence and scale rules (absolute slack 0.05, plus unit/scale factors).

A post-run **stray-number audit** was performed. A prediction was reclassified as incorrect when:

1. the official full-writeup scorer marked it correct,
2. the final stated answer did not match the gold answer, and
3. a non-final intermediate quantity was responsible for the match.

Predictions were **not regenerated**. The evaluator code was **not modified**. The corrected IDs are recorded in each run's `evaluation.json` under `stray_number_reclassified`. The audit affected only a small number of official-correct predictions and did not change any model predictions or computational measurements.

Reclassified IDs: Full SR `tatqa_test_9162`; NEED/REL/SUP `tatqa_test_9180`; USEFUL `tatqa_test_9180`, `finqa_test_927`, `tatqa_test_9642`. Per-question 0/1 matrix: `experiments/self_rag_ablation_correctness_matrix.csv`.

---

## Evaluation Sets

| Set               | Path                                              | Role                                                              |
| ------------------ | -------------------------------------------------- | ------------------------------------------------------------------ |
| Development 15-Q   | `data/t2_ragbench/questions/t2_small_15.json`      | Plumbing / sanity (5 FinQA + 5 ConvFinQA + 5 TAT-DQA)              |
| Frozen 120-Q       | `experiments/final_120_questions.json`              | Final comparison set; 15-Q ids excluded; do not resample. 40 FinQA / 40 ConvFinQA / 40 TAT-DQA, 120 documents, 100 companies |

Hugging Face source: `G4KMU/t2-ragbench`. The adapter downloads the page PDFs needed for a run and indexes them with the frozen pipeline. This set was used for plumbing and sanity checks only and is **not** used for the final 120-question comparison. The final manifest was frozen before the final experiments and was unchanged throughout the comparison.

---

## Development 15-Q Results

Lumer baseline with query formulation (`logs/t2-ragbench-small-queryform_20260823T045242Z/`):

| Metric                    | Value |
| -------------------------- | ----- |
| Accuracy                   | 8/15 = 53.3% |
| Recall@1 / @3 / @5 / @10    | 1.0   |
| MRR                         | 1.0   |
| Mean LLM calls              | 2.0   |
| Rank-1 gold document        | 15/15 |

Retrieval matched the earlier 15-Q conventional baseline (query = question) at Recall@1 = 1.0 / MRR = 1.0. ConvFinQA remains 0/5 on this small set (insufficient-evidence answers despite gold-document hits).

Do not treat `logs/t2-ragbench-final-120_20260822T120619Z/` (43/120 = 35.8%, query = question, 1 LLM call) as the Lumer Agentic RAG baseline.

Self-RAG V2 15-Q (window refine; original-question retrieval, before formulation was added): 13/15 accuracy, Recall@1 = 1.0. That number is not the fair 120-Q comparison, since the current SYSTEM B now uses the QueryFormulator.

---

## Final Experimental Results (Frozen 120-Q)

All final systems use the same 120-question manifest, document corpus, retrieval index, QueryFormulator, hybrid retrieval configuration, Qwen generator, and evaluation protocol.

The reported accuracies below include the post-run stray-number audit (raw scorer value in parentheses where it differed). Lumer had no stray-number cases.

| System                | Accuracy            | FinQA | ConvFinQA | TAT-DQA | Latency (s) | Tokens  | LLM calls | Evidence refinements | Log |
| ---------------------- | -------------------- | ----- | --------- | ------- | ------------ | ------- | --------- | --------------------- | --- |
| **Lumer Agentic RAG**   | **45/120 = 37.5%**   | 12/40 | 15/40     | 18/40   | 18.97        | 1269.7  | 2.00      | —                      | `logs/t2-ragbench-final-120-agentic_20260823T050108Z/` |
| **Full Self-RAG V2**    | **61/120 = 50.8%** (raw 62) | 16/40 | 19/40 | 26/40 | 56.38 | 5762.2 | 7.49 | 0.892 | `logs/t2-self-rag-final-120-agentic_20260823T054650Z/` |
| NEED_RETRIEVAL OFF      | 58/120 = 48.3% (raw 59) | 15/40 | 19/40 | 24/40 | 54.91 | 5951.2 | 6.73 | 0.875 | `logs/t2-self-rag-ablate-need-retrieval_20260823T080227Z/` |
| RELEVANT OFF            | 60/120 = 50.0% (raw 61) | 16/40 | 19/40 | 25/40 | 50.75 | 5196.0 | 6.79 | 0.875 | `logs/t2-self-rag-ablate-relevant_20260823T095300Z/` |
| SUPPORTED OFF           | 59/120 = 49.2% (raw 60) | 16/40 | 19/40 | 24/40 | 43.93 | 4265.5 | 5.97 | 0.858 | `logs/t2-self-rag-ablate-supported_20260823T113537Z/` |
| USEFUL OFF              | 57/120 = 47.5% (raw 60) | 16/40 | 18/40 | 23/40 | 36.17 | 3792.5 | 5.66 | 0.658 | `logs/t2-self-rag-ablate-useful_20260823T130730Z/` |

Drop vs Full Self-RAG (61): NEED −3, REL −1, SUP −2, USE −4. Largest drop: USEFUL OFF after the audit (NEED was largest on the raw scorer).

---

## Main Result

```text
Lumer Agentic RAG
45/120 = 37.5%

Full Self-RAG V2
61/120 = 50.8%

Improvement
+13.3 percentage points
```

The two systems use identical retrieval conditions. Both obtain:

```text
Recall@1 = 0.892
Recall@5 = 0.950
MRR      = 0.917
```

Therefore, the observed accuracy improvement occurs without an improvement in retrieval metrics. This supports the interpretation that the improvement primarily arises from **post-retrieval reflection and evidence use** rather than from retrieving more relevant documents.

---

## Dataset-Level Results

| Dataset   | Lumer Agentic RAG | Full Self-RAG V2 | Improvement |
| --------- | ------------------ | ------------------ | ------------ |
| FinQA     | 30.0%               | **40.0%**          | +10.0 pp     |
| ConvFinQA | 37.5%               | **47.5%**          | +10.0 pp     |
| TAT-DQA   | 45.0%               | **65.0%**          | +20.0 pp     |

The improvement is observed across all three datasets, with the largest observed gain on TAT-DQA.

---

## Paired Correctness Analysis

At the question level, comparing the audited labels:

```text
Lumer wrong  -> Self-RAG correct: 21
Lumer correct -> Self-RAG wrong:   5
```

Thus Self-RAG produces a net improvement of:

```text
21 - 5 = 16 questions
```

---

## Ablation Study

The four one-component ablations evaluate the contribution of individual reflection stages.

| Configuration        | Accuracy | Change vs Full Self-RAG |
| ---------------------- | -------- | ------------------------ |
| **Full Self-RAG V2**    | **50.8%** | —                        |
| NEED_RETRIEVAL OFF      | 48.3%    | −2.5 pp                  |
| RELEVANT OFF            | 50.0%    | −0.8 pp                  |
| SUPPORTED OFF           | 49.2%    | −1.6 pp                  |
| USEFUL OFF              | **47.5%** | **−3.3 pp**              |

Every individual ablation performs below the complete Self-RAG system. The largest observed accuracy reduction occurs when **USEFUL** is removed, suggesting that usefulness assessment makes the largest individual contribution to accuracy among the four tested components, although the experiment does not establish statistical significance between individual ablation effects.

---

## Accuracy–Efficiency Analysis

Full Self-RAG provides the highest accuracy but also requires substantially more inference computation.

### Full Self-RAG vs USEFUL OFF

| Metric                    | Full Self-RAG | USEFUL OFF |
| -------------------------- | -------------- | ---------- |
| Accuracy                   | **50.8%**       | 47.5%      |
| Mean latency                | 56.38 s         | 36.17 s    |
| Mean tokens                  | 5762.2          | 3792.5     |
| Mean LLM calls               | 7.49            | 5.66       |
| Mean evidence refinements    | 0.892           | 0.658      |

Keeping USEFUL produces +3.3 percentage points accuracy at the cost of substantially greater inference time, token usage, and LLM calls.

The relationship between computation and accuracy is **not strictly monotonic across all ablations**. For example, NEED_RETRIEVAL OFF has relatively high computational cost but lower accuracy than the complete system. This indicates that different reflection components have different computational efficiency.

---

## Qualitative Error Analysis

Observed ablation-sensitive errors include:

- insufficient-evidence abstention
- selecting a nearby but incorrect financial value
- selecting the wrong table row or metric
- incorrect direction/sign in numerical differences
- failure to refine evidence when additional context is required

The majority of manually inspected ablation-sensitive flips were associated with **table-heavy financial questions**, where multiple nearby numerical values can make evidence selection difficult.

Question-type classifications used for this analysis are **inferred from the frozen manifest classifier**, rather than native dataset annotations.

---

## Experimental Logs

Each experiment creates a new timestamped directory (never overwrite older 15-Q or 120-Q folders):

```text
logs/<run_name>_<UTC>/
  {question_id}.json
  records.jsonl
  evaluation.json
  run_summary.json
```

Baseline records include original question, formulated query, formulation prompt/raw output/latency/tokens, filters, chunk ids/ranks/scores, answer, `number_of_llm_calls` (2), and generation latency/tokens.

The frozen 120-Q Lumer, Full Self-RAG, and four ablation directories are checked in. Older small/baseline folders stay gitignored.

Final frozen runs:

```text
logs/t2-ragbench-final-120-agentic_20260823T050108Z/
logs/t2-self-rag-final-120-agentic_20260823T054650Z/
logs/t2-self-rag-ablate-need-retrieval_20260823T080227Z/
logs/t2-self-rag-ablate-relevant_20260823T095300Z/
logs/t2-self-rag-ablate-supported_20260823T113537Z/
logs/t2-self-rag-ablate-useful_20260823T130730Z/
```

The per-question correctness matrix is available at `experiments/self_rag_ablation_correctness_matrix.csv`.

---

## Reproducing the Experiments

### Installation

```bash
python3 -m pip install -r requirements.txt
```

Start Ollama with `qwen3:4b-instruct-2507-q4_K_M` on `localhost:11434`. The first embedding run downloads `BAAI/bge-small-en-v1.5` from Hugging Face.

Run the test suite:

```bash
python3 -m pytest tests -q
```

### Development 15-Q Run (existing small index)

```bash
python3 experiments/run_t2_ragbench.py \
  --questions data/t2_ragbench/questions/t2_small_15.json
```

### Frozen 120-Q Lumer Agentic RAG Baseline

*(Do not rebuild the index if `data/indexes/t2_ragbench_final120` already exists.)*

```bash
python3 experiments/run_t2_ragbench.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-ragbench-final-120-agentic
```

`--pdf-dir` / `--index-dir` / `--run-name` are existing runner flags so the 120-Q corpus is used and old conventional-RAG logs are not overwritten.

### Frozen 120-Q Full Self-RAG

*(Same questions/index; do not rebuild the index.)*

```bash
python3 experiments/run_t2_self_rag.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-self-rag-final-120-agentic
```

### Ablations

The four component ablations use the same frozen 120-question set and retrieval index (sequential; do not overwrite old log dirs).

**NEED_RETRIEVAL OFF**

```bash
python3 experiments/run_t2_self_rag.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-self-rag-ablate-need-retrieval \
  --no-retrieval-decision
```

**RELEVANT OFF**

```bash
python3 experiments/run_t2_self_rag.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-self-rag-ablate-relevant \
  --no-retrieval-relevance
```

**SUPPORTED OFF**

```bash
python3 experiments/run_t2_self_rag.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-self-rag-ablate-supported \
  --no-evidence-support
```

**USEFUL OFF**

```bash
python3 experiments/run_t2_self_rag.py \
  --questions experiments/final_120_questions.json \
  --pdf-dir data/t2_ragbench/documents_final120 \
  --index-dir data/indexes/t2_ragbench_final120 \
  --run-name t2-self-rag-ablate-useful \
  --no-answer-usefulness
```

### Tiny checked-in sample corpus

A tiny checked-in sample (Northstar / CedarTech) is still available via `experiments/build_sample_corpus.py` and `experiments/run_baseline.py`.

---

## Development vs Final Evaluation

The 15-question development experiment is intended only for plumbing, debugging, sanity checking, and early development. It must not be mixed with the final 120-question comparison.

The older conventional-RAG result (`43/120 = 35.8%`) is **not** the Lumer Agentic RAG baseline because it used the original question directly as the retrieval query and one LLM call. The final Lumer baseline instead uses the Lumer-style QueryFormulator and two LLM calls.

---

## Limitations

- The final evaluation uses a frozen **120-question subset** rather than the complete T²-RAGBench benchmark.
- Each dataset contributes **40 questions**, limiting statistical power for dataset-specific conclusions.
- The reflection pipeline introduces substantial inference cost.
- The qualitative question-type classifications are inferred rather than dataset-native annotations.
- Answer evaluation required a post-run stray-number audit because the original scorer could mark a free-form answer correct based on an intermediate numerical quantity.
- The study evaluates one frozen local Qwen generator and one fixed retrieval configuration.

These limitations should be considered when generalizing the results beyond the evaluated setting.

---

## Tests

```bash
python3 -m pytest tests -q
```

The final experimental runs were performed only after the test suite passed.

---

## Dataset

The experiments use **T²-RAGBench** (`G4KMU/t2-ragbench`), containing financial QA data from FinQA, ConvFinQA, and TAT-DQA. The adapter downloads the page PDFs required for the frozen evaluation corpus.

---

## Citation

Paper forthcoming. Once the paper is published or a preprint is available, the citation and DOI will be added here.

---

## License

License information will be added before public release. Please verify the redistribution terms of the underlying datasets, filing PDFs, pretrained models, and third-party dependencies before redistributing those artifacts.
