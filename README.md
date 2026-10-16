# Clinical Speech Evaluation Agent

A multi-agent RAG system that automates clinical speech evaluation using **LangGraph**, **ChromaDB** and **OpenAI**. Built as a portfolio project demonstrating RAG pipelines, multi-agent orchestration, hybrid search, agent memory and human-in-the-loop review.

---

## What It Does

Takes pairs of **reference transcripts** (what was said) and **recognized transcripts** (what the speech model heard), evaluates recognition quality using clinical context, and produces a structured report — automatically comparing against previous runs to detect regressions.

```
DataPrepAgent → RAGIndexingAgent → EvaluationAgent → HumanReviewNode → ReportAgent
                                         ↓
                              (if no borderline cases)
                                     ReportAgent
```

| Agent | Responsibility |
|---|---|
| **DataPrepAgent** | Loads dataset, removes duplicate transcript pairs |
| **RAGIndexingAgent** | Embeds clinical KB documents into ChromaDB |
| **EvaluationAgent** | Calculates WER, CER, exact match; retrieves clinical context for failures |
| **HumanReviewNode** | Pauses graph for human approval of borderline cases (LangGraph interrupt) |
| **ReportAgent** | Generates structured report; compares against previous runs from memory |

---

## Project Structure

```
clinical-speech-eval-agent/
├── main.py                          # Entry point
├── requirements.txt
├── .env.example
├── data/
│   ├── generate_synthetic_data.py   # Generates synthetic clinical dataset
│   └── synthetic_transcripts.json  # Generated dataset (created at runtime)
├── src/
│   ├── agents/
│   │   ├── data_prep_agent.py       # Deduplication and data loading
│   │   ├── rag_agent.py             # ChromaDB indexing
│   │   ├── evaluation_agent.py      # WER/CER/exact-match scoring + RAG context
│   │   └── report_agent.py          # Report generation and regression comparison
│   ├── graph/
│   │   ├── state.py                 # LangGraph EvalState TypedDict
│   │   └── workflow.py              # StateGraph definition, nodes, edges
│   ├── rag/
│   │   ├── embeddings.py            # ChromaDB client, index, hybrid search
│   │   └── reranker.py              # Cross-encoder reranking
│   ├── evaluation/
│   │   └── metrics.py               # WER, CER, exact match, classification
│   ├── memory/
│   │   └── run_store.py             # Persist and compare evaluation runs
│   └── utils/
│       └── human_review.py          # Human-in-the-loop CLI handler
├── tests/
│   ├── test_metrics.py
│   ├── test_rag.py
│   └── test_agents.py
└── output/                          # Generated reports (created at runtime)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph, MemorySaver, interrupt) |
| LLM | OpenAI GPT-4o |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Reranker | cross-encoder `ms-marco-MiniLM-L-6-v2` |
| Speech metrics | jiwer (WER, CER) |
| Data processing | Python, Pandas |
| Testing | pytest |

---

## Prerequisites

- Python 3.11 or higher
- An OpenAI API key ([platform.openai.com](https://platform.openai.com))
- pip

---

## Setup and Run — Step by Step

### Step 1 — Clone or navigate to the project

```bash
cd "/Users/I752029/Library/CloudStorage/OneDrive-SAPSE/Desktop/Learnings/clinical-speech-eval-agent"
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `langgraph`, `langchain`, `langchain-openai`, `chromadb`, `jiwer`, `sentence-transformers`, `pandas`, `pytest`, `python-dotenv`

> First run downloads sentence-transformer and cross-encoder models (~100 MB). Subsequent runs use the local cache.

### Step 4 — Add your OpenAI API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-...your-key-here...
```

### Step 5 — Generate the synthetic dataset

```bash
python data/generate_synthetic_data.py
```

Expected output:
```
Generated 61 transcript pairs and 15 KB documents → data/synthetic_transcripts.json
  clean: 24
  borderline: 18
  failed: 18
  (1 duplicate included to test dedup)
```

### Step 6 — Run the full pipeline

```bash
python main.py
```

Expected output:
```
============================================================
  Clinical Speech Evaluation Agent
============================================================
[DataPrepAgent] Loaded 61 pairs → 60 after dedup (1 duplicate removed). KB: 15 docs.
[RAGAgent] Indexed 15 new KB document(s) into ChromaDB.
[EvaluationAgent] Evaluating 60 transcript pairs...
[EvaluationAgent] Results: {'clean': 24, 'borderline': 18, 'failed': 18}. Borderline cases: 18

[HumanReview] 18 borderline case(s) require review:
----------------------------------------------------------------------
  ID       : pair_005  (NICU)
  Reference: Patient transferred from NICU following stabilization of ...
  Recognized: Patient transferred from NICO following stabilization of ...
  WER: 25.00%  CER: 8.33%
  Context  : NICU Overview | NICU — Respiratory Support Terminology
  ...

[main] Graph paused for human review.
       (Auto-approving all borderline cases for demo.)

[ReportAgent] Run run_20261013_200512 complete.
  Total: 60 | Clean: 24 | Borderline: 18 | Failed: 18
  Avg WER: 31.24% | Avg CER: 18.45% | Exact Match: 24/60
  Report saved → output/run_20261013_200512.json
```

### Step 7 — Run a second time (tests regression detection)

```bash
python main.py --thread-id session-2
```

The report now includes a comparison against the previous run:
```
  vs run_20261013_200512: WER delta +0.00% | Regressions: 0 | Improvements: 0
```

### Step 8 — Run the tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_metrics.py::test_wer_perfect_match          PASSED
tests/test_metrics.py::test_wer_complete_mismatch      PASSED
tests/test_metrics.py::test_exact_match_case_insensitive PASSED
tests/test_rag.py::test_semantic_search_relevance      PASSED
tests/test_rag.py::test_semantic_search_category_filter PASSED
tests/test_agents.py::test_data_prep_removes_duplicates PASSED
...
15 passed in X.XXs
```

---

## Using Custom Data

Replace `data/synthetic_transcripts.json` with your own file following this schema:

```json
{
  "transcript_pairs": [
    {
      "id": "pair_001",
      "audio_file": "recording_001.wav",
      "category": "NICU",
      "reference": "Patient transferred from NICU",
      "recognized": "Patient transferred from NICO",
      "expected_wer_range": "borderline"
    }
  ],
  "knowledge_base": [
    {
      "id": "kb_001",
      "category": "NICU",
      "title": "NICU Overview",
      "content": "NICU stands for Neonatal Intensive Care Unit..."
    }
  ]
}
```

Then run:

```bash
python main.py --data path/to/your/data.json
```

---

## Output

Each run produces a JSON report in `output/`:

```json
{
  "run_id": "run_20261013_200512",
  "timestamp": "2026-10-13T20:05:12",
  "summary": {
    "total_pairs": 60,
    "clean": 24,
    "borderline": 18,
    "failed": 18,
    "avg_wer": 0.3124,
    "avg_cer": 0.1845,
    "exact_matches": 24,
    "exact_match_rate": 0.4
  },
  "run_comparison": {
    "previous_run_id": "run_20261013_195501",
    "avg_wer_delta": 0.0,
    "regressions": [],
    "improvements": []
  },
  "results": [ ... ]
}
```

Run history is stored in `output/runs_history.json` and used automatically for regression detection on every subsequent run.

---

## Key Concepts Demonstrated

**RAG Pipeline** — Clinical KB documents are chunked, embedded with `all-MiniLM-L6-v2` and stored in ChromaDB. On evaluation failures the agent retrieves relevant context using hybrid search (semantic + metadata category filter) and reranks with a cross-encoder.

**Multi-Agent Orchestration** — LangGraph `StateGraph` connects five agents in a conditional pipeline. State is shared across all agents via a typed `EvalState` dictionary.

**Agent Memory** — `MemorySaver` checkpointer persists LangGraph state across graph invocations. `run_store.py` persists evaluation history to disk for cross-run regression detection.

**Human-in-the-Loop** — LangGraph `interrupt()` pauses the graph after evaluation, surfaces borderline cases for human review, then resumes when decisions are provided.

**Speech Metrics** — WER and CER computed with `jiwer`. Results classified as clean (WER ≤ 20%), borderline (20–50%) or failed (> 50%).
