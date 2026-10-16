# Understanding the Project — Clinical Speech Evaluation Agent

This document explains what the project does, why each piece exists, and how everything connects. It also includes a quick-start guide so you can run everything without switching files.

---

## Quick Start — How to Run

### 1. Install dependencies (one time)
```bash
cd "/clinical-speech-eval-agent"
pip install -r requirements.txt
```

### 2. Add your OpenAI key (one time)
```bash
cp .env.example .env
# open .env and set: OPENAI_API_KEY=sk-...your-key...
```

### 3. Generate the dataset (one time)
```bash
python data/generate_synthetic_data.py
```

### 4. Run — choose one

**Option A — UI (recommended)**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Use the **Dashboard** page to run the pipeline and view results. Use the **Dataset Builder** page to create your own transcript pairs.

**Option B — Command line**
```bash
python main.py                                  # run with synthetic data
python main.py --data data/my_dataset.json      # run with your own data
python main.py --thread-id session-2            # second run (triggers regression check)
```

### 5. Run tests
```bash
pytest tests/ -v       # expects 23 passed
```

> For the full setup guide with troubleshooting, see [INSTRUCTIONS.md](INSTRUCTIONS.md).

---

## What problem does this solve?

Clinical environments — hospital wards, ICUs, emergency departments — use voice-to-text software to capture spoken notes. These systems often mishear medical acronyms: a nurse says "NICU" (Neonatal Intensive Care Unit) and the model transcribes "NICO", "NIK U", or "PEE KU".

Evaluating these errors manually across hundreds of transcript pairs is slow. This project automates that evaluation using a multi-agent AI pipeline that:

1. Loads and cleans a batch of transcript pairs
2. Scores each pair using speech-recognition metrics (WER, CER)
3. Retrieves clinical context for ambiguous errors using RAG
4. Pauses for a human to review borderline cases
5. Generates a structured report and compares it against previous runs

---

## Core concepts you need to know

### Word Error Rate (WER)

WER measures how wrong a transcription is at the word level.

```
Reference : NICU team performed rounds
Recognized: NICO team performed rounds

1 word wrong out of 4 → WER = 25%
```

This project classifies results into three buckets:

| WER | Classification | Meaning |
|---|---|---|
| 0 – 20% | **clean** | Acceptable recognition |
| 20 – 50% | **borderline** | Possibly acceptable, needs human review |
| > 50% | **failed** | Significant error, likely unusable |

### Character Error Rate (CER)

CER measures errors at the character level instead of the word level. Useful for catching partial errors within a word:

```
NICU → NICO   (1 char wrong out of 4 → CER = 25%)
NICU → NIK U  (2 chars wrong → CER = 50%)
```

CER is reported alongside WER but is not used for classification.

### RAG (Retrieval-Augmented Generation)

For borderline and failed cases, the pipeline retrieves relevant clinical documents to give context. Example: if a NICU transcript fails, the system fetches the "NICU — Neonatal Intensive Care Unit" and "NICU — Respiratory Support Terminology" knowledge-base documents and attaches them to the result.

This shows reviewers *why* an acronym matters clinically and what the correct usage is.

---

## Architecture — how the agents connect

```
Input JSON
    │
    ▼
┌─────────────────┐
│  DataPrepAgent  │  Loads pairs, removes duplicates by MD5 hash
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAGAgent       │  Embeds clinical KB documents into ChromaDB
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EvaluationAgent │  Scores every pair (WER, CER, exact match)
│                 │  Retrieves clinical context for non-clean pairs
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  Has       No
borderline borderline
    │         │
    ▼         │
┌──────────┐  │
│ HumanReview│ │  Pauses graph, shows cases, applies decisions
└────┬─────┘  │
     │        │
     └───┬────┘
         │
         ▼
┌─────────────────┐
│  ReportAgent    │  Builds JSON report, compares vs previous run
└─────────────────┘
         │
         ▼
  output/run_*.json
```

The agents share state through a single `EvalState` dictionary that flows through the entire graph. No agent calls another directly — they only read from and write to this shared state.

---

## Each agent explained

### DataPrepAgent (`src/agents/data_prep_agent.py`)

Loads the dataset JSON and removes duplicate transcript pairs. Duplicates are detected by computing an MD5 hash of `"reference|recognized"` — if two pairs produce the same hash, only the first is kept.

Why this matters: real datasets often have near-identical entries from transcription re-runs. Processing duplicates inflates metrics and wastes computation.

### RAGAgent (`src/agents/rag_agent.py`)

Takes the knowledge-base documents from the dataset and indexes them into ChromaDB. Each document is converted into a vector embedding using the `all-MiniLM-L6-v2` model (a lightweight sentence transformer).

ChromaDB stores these vectors in memory and supports cosine-similarity search. If the same documents are indexed twice, the upsert operation skips them (dedup by document ID).

### EvaluationAgent (`src/agents/evaluation_agent.py`)

The core of the pipeline. For every transcript pair:

1. Calculates WER, CER, and exact match
2. Classifies the result as clean / borderline / failed
3. For non-clean pairs only: queries ChromaDB for the top-3 most relevant KB documents, then reranks them with a cross-encoder to get the top 2
4. Appends the result to the evaluation list

The reranker (`ms-marco-MiniLM-L-6-v2`) is a cross-encoder that scores each retrieved document against the query directly — it is more accurate than cosine similarity alone but slower, so it only runs on borderline/failed cases.

### HumanReviewNode (`src/utils/human_review.py`)

Uses LangGraph's `interrupt()` function to pause the entire graph mid-execution. The graph checkpoints its state to memory (via `MemorySaver`), and execution stops until a decision is provided.

In demo mode (`main.py`), all borderline cases are auto-approved. In production use, a human would provide a dictionary of `{pair_id: "approve" | "reject" | "modify:<text>"}` to the graph's resume call.

This is a genuine pause — not a print statement followed by a question. The process actually halts and the state is serialized to disk (in a real deployment, this could be a database, allowing multi-day review workflows).

### ReportAgent (`src/agents/report_agent.py`)

Aggregates all results into a summary report and compares against the previous run:

- If a pair's WER increased by more than 5% vs the previous run → **regression**
- If it decreased by more than 5% → **improvement**

The comparison data is saved to `output/runs_history.json`. This file is append-only and grows with every run, enabling trend analysis over time.

---

## The RAG pipeline in detail

```
Query: "CVICU SEE VIK YOU admission at zero three hundred hours"
           │
           ▼
     ChromaDB cosine search
     (top 3 by vector similarity)
           │
           ▼
   [CVICU — Cardiovascular ICU]  score 0.87
   [CVICU — Hemodynamic Monitoring]  score 0.82
   [CVICU — Post-Cardiac Surgery Care]  score 0.79
           │
           ▼
   Cross-encoder reranker
   (scores each doc against the query directly)
           │
           ▼
   Top 2: CVICU — Cardiovascular ICU | CVICU — Hemodynamic Monitoring
```

The two-stage approach exists because cosine similarity on embeddings is fast but approximate — it finds semantically similar vectors but can miss precise relevance. The cross-encoder is slower but reads the full query and document together, giving a more accurate score. Running it only on the top-3 cosine hits keeps latency manageable.

The `category_filter` parameter in `semantic_search()` adds a metadata filter to ChromaDB so that a NICU query only retrieves NICU documents, not ICU or PICU ones.

---

## LangGraph — why not just write functions?

A plain Python script calling functions in sequence would work for the happy path. LangGraph adds three things this project needs:

1. **Checkpointing** — `MemorySaver` serializes graph state after every node. If the process crashes mid-run, you can resume from the last checkpoint.

2. **Human-in-the-loop** — `interrupt()` pauses execution at an arbitrary node and waits for external input. Without LangGraph, implementing this would require a message queue, a separate process, and custom state serialization.

3. **Conditional routing** — the edge from `EvaluationAgent` to either `HumanReviewNode` or `ReportAgent` is a function, not hardcoded. This makes it easy to add more routing logic later (e.g., skip human review if all WERs are below 25%).

---

## Agent memory — two levels

This project has memory at two levels:

| Level | Mechanism | Scope |
|---|---|---|
| **Within a run** | LangGraph `MemorySaver` checkpointer | State persists across the interrupt/resume cycle within one pipeline execution |
| **Across runs** | `src/memory/run_store.py` → `output/runs_history.json` | Results from every run are written to disk, enabling regression detection between separate executions |

The `MemorySaver` is tied to a `thread_id`. Running `python main.py --thread-id session-2` starts a fresh LangGraph thread that does not share in-memory state with `session-1`, but both threads read from the same `runs_history.json`.

---

## Data flow — what moves through the graph

```python
EvalState = {
    "dataset_path": str,          # set at entry point
    "raw_pairs": [...],            # set by DataPrepAgent
    "clean_pairs": [...],          # set by DataPrepAgent (after dedup)
    "kb_documents": [...],         # set by DataPrepAgent
    "eval_results": [...],         # set by EvaluationAgent
    "borderline_ids": [...],       # set by EvaluationAgent
    "approved_results": [...],     # set by HumanReviewNode
    "run_comparison": {...},       # set by ReportAgent
    "final_report": {...},         # set by ReportAgent
    "error": str | None,           # set if any node raises
}
```

Each agent receives the full state dict and returns only the keys it changed. LangGraph merges the return value back into the state before passing it to the next node.

---

## File structure map

```
clinical-speech-eval-agent/
│
├── main.py                      Entry point — builds graph, streams events, handles HITL
├── app.py                       Streamlit dashboard + Dataset Builder UI
│
├── data/
│   ├── generate_synthetic_data.py   Produces synthetic_transcripts.json
│   └── synthetic_transcripts.json   Generated dataset (61 pairs + 15 KB docs)
│
├── src/
│   ├── graph/
│   │   ├── state.py             EvalState TypedDict — the shared data contract
│   │   └── workflow.py          StateGraph: nodes, edges, conditional routing
│   │
│   ├── agents/
│   │   ├── data_prep_agent.py   Load + deduplicate transcript pairs
│   │   ├── rag_agent.py         Index KB into ChromaDB
│   │   ├── evaluation_agent.py  WER/CER/exact-match + RAG context retrieval
│   │   └── report_agent.py      Aggregate metrics + regression comparison + save JSON
│   │
│   ├── rag/
│   │   ├── embeddings.py        ChromaDB client, index_documents, semantic_search
│   │   └── reranker.py          Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   │
│   ├── evaluation/
│   │   └── metrics.py           calculate_wer, calculate_cer, exact_match, classify_result
│   │
│   ├── memory/
│   │   └── run_store.py         load_latest_run / save_run → runs_history.json
│   │
│   └── utils/
│       └── human_review.py      human_review_node — interrupt, display, apply decisions
│
├── tests/
│   ├── test_metrics.py          14 unit tests: WER/CER/exact-match/classify boundaries
│   ├── test_rag.py              5 tests: index dedup, semantic search, category filter
│   └── test_agents.py           3 tests: dedup logic, unique pair retention, KB loading
│
└── output/
    ├── run_YYYYMMDD_HHMMSS.json   One report per run
    └── runs_history.json          Cross-run history for regression detection
```

---

## Output report — what each field means

```json
{
  "run_id": "run_20261013_212415",
  "timestamp": "2026-10-13T21:24:15",
  "summary": {
    "total_pairs": 60,
    "clean": 30,          ← WER ≤ 20%
    "borderline": 22,     ← WER 20–50%, went through human review
    "failed": 1,          ← WER > 50%
    "avg_wer": 0.1682,    ← mean WER across all pairs (16.82%)
    "avg_cer": 0.0369,    ← mean CER across all pairs
    "exact_matches": 17,  ← pairs where normalized text is identical
    "exact_match_rate": 0.3208
  },
  "run_comparison": {
    "previous_run_id": "run_20261013_195501",
    "avg_wer_delta": 0.0,   ← positive = got worse, negative = improved
    "regressions": [],       ← pair IDs where WER increased > 5%
    "improvements": []       ← pair IDs where WER decreased > 5%
  },
  "results": [
    {
      "id": "pair_009",
      "category": "NICU",
      "reference": "NICU team performed morning rounds...",
      "recognized": "NICO team performed morning rounds...",
      "wer": 0.2222,
      "cer": 0.0536,
      "exact_match": false,
      "classification": "borderline",
      "clinical_context": "NICU — Neonatal Intensive Care Unit | NICU — Respiratory Support Terminology",
      "human_decision": "approved"  ← set after human review
    }
  ]
}
```

---

## Tech stack — why each tool was chosen

| Tool | Role | Why this one |
|---|---|---|
| **LangGraph** | Agent orchestration | Native support for interrupt/resume (HITL), state checkpointing, conditional edges |
| **LangChain** | LLM abstraction layer | Standardized interface for OpenAI calls |
| **OpenAI GPT-4o** | LLM | High accuracy on clinical text; used in report generation |
| **ChromaDB** | Vector store | Runs entirely in-memory (no server setup), fast for small-to-medium datasets |
| **sentence-transformers all-MiniLM-L6-v2** | Embeddings | Lightweight (80 MB), fast inference, good semantic accuracy for English text |
| **cross-encoder ms-marco-MiniLM-L-6-v2** | Reranker | Purpose-trained for passage ranking, significantly improves retrieval precision |
| **jiwer** | WER/CER metrics | Industry-standard Python library for speech recognition evaluation |
| **Streamlit** | UI | Pure Python, no frontend skills needed, fast to build data dashboards |
| **Plotly** | Charts | Interactive charts with hover, filter, zoom built in |

---

## Key design decisions

**Why interrupt() instead of asking for input with input()?**
`input()` blocks the entire process and cannot be serialized. LangGraph's `interrupt()` serializes the graph state to the checkpointer, which in production could be a Postgres database. The reviewer could come back hours later from a different machine and resume the exact same graph run.

**Why separate WER and CER?**
WER penalizes "NICU → P I C U" heavily (4 words become 4 words but all wrong). CER would score this more leniently since many characters are shared. Reporting both gives reviewers a richer picture: a high WER with a low CER suggests the model spelled the concept correctly but split or merged tokens.

**Why MD5 hash for dedup?**
Computing the hash of `"reference|recognized"` is O(n) and requires no database. Comparing every pair against every other pair would be O(n²). For a dataset of thousands of pairs this matters; for 60 pairs it does not, but the pattern scales.

**Why auto-approve in demo mode?**
The pipeline is a portfolio project. Running it in a terminal requires no human sitting at the keyboard waiting to type decisions. Auto-approve lets you see the full pipeline end to end in one command. In a real deployment you would replace auto-approve with a web form, a Slack notification, or any other review mechanism.
