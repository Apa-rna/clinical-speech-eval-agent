# How to Run — Clinical Speech Evaluation Agent

This guide covers every way to run the project: the command-line pipeline, the Streamlit UI, and the test suite. No prior knowledge of the project is assumed.

---

## Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | 3.11 or higher | f-string and type-hint features |
| pip | any recent version | install dependencies |
| OpenAI API key | — | GPT-4o calls in the pipeline |

Get an API key at [platform.openai.com](https://platform.openai.com) → API Keys → Create new secret key.

---

## Step 1 — Open the project folder

```bash
cd "/Users/I752029/Library/CloudStorage/OneDrive-SAPSE/Desktop/Learnings/clinical-speech-eval-agent"
```

All commands below assume you are inside this folder.

---

## Step 2 — Install dependencies (one time only)

```bash
pip install -r requirements.txt
```

This installs: `langgraph`, `langchain`, `langchain-openai`, `chromadb`, `jiwer`, `sentence-transformers`, `pandas`, `streamlit`, `plotly`, `pytest`, `python-dotenv`.

The first run also downloads two AI models (~100 MB total) from Hugging Face:
- `all-MiniLM-L6-v2` — for embedding clinical documents
- `ms-marco-MiniLM-L-6-v2` — for reranking search results

These are cached locally after the first download.

---

## Step 3 — Add your OpenAI API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-...your-key-here...
```

Save and close the file. The pipeline reads this automatically.

---

## Step 4 — Generate the synthetic dataset (one time only)

```bash
python data/generate_synthetic_data.py
```

This creates `data/synthetic_transcripts.json` with:
- 61 transcript pairs across 6 clinical categories (NICU, CVICU, PICU, ICU, HUC, ED)
- 15 clinical knowledge-base documents

Expected output:
```
Generated 61 transcript pairs and 15 KB documents → data/synthetic_transcripts.json
  clean: 25
  borderline: 18
  failed: 18
  (1 duplicate included to test dedup)
```

---

## Option A — Run via the UI (recommended)

The Streamlit dashboard lets you run the pipeline, build datasets, and explore results visually.

```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

### Dashboard page

- Sidebar shows **Run Pipeline** button — click to run the pipeline on the synthetic dataset
- Sidebar dropdown lets you switch between past report files
- Four tabs:
  - **Overview** — KPI cards, donut chart, WER by category, WER distribution histogram
  - **All Results** — filterable color-coded table (green = clean, yellow = borderline, red = failed)
  - **Borderline & Failed** — expandable cards with side-by-side reference vs recognized text
  - **Run History** — WER trend line and classification stacked bar (visible after 2+ runs)

### Dataset Builder page

Switch to **Dataset Builder** in the sidebar to create your own dataset.

1. **Load existing** — pick a `.json` file from the `data/` folder, or upload your own
2. **Add pair** — fill in Category, Reference text, Recognized text; ID is auto-generated
3. **Edit table** — the grid is fully editable: click any cell to change it, use the bottom row to add more
4. **Delete** — expand "Delete a pair" to remove one row by ID
5. **Save dataset** — writes a `.json` file to `data/` that you can reload later
6. **Save & Run pipeline** — saves then immediately runs the pipeline; output log appears on screen

After running from the Dataset Builder, switch to **Dashboard** to see the results.

---

## Option B — Run via the command line

### Run with the synthetic dataset

```bash
python main.py
```

### Run with a custom dataset

```bash
python main.py --data data/my_dataset.json
```

### Run a second time (regression detection)

```bash
python main.py --thread-id session-2
```

Changing `--thread-id` starts a fresh LangGraph session. The report will automatically compare against the previous run's metrics.

### What the terminal output looks like

```
============================================================
  Clinical Speech Evaluation Agent
============================================================
[DataPrepAgent] Loaded 61 pairs → 60 after dedup (1 duplicate removed). KB: 15 docs.
[RAGAgent] Indexed 15 new KB document(s) into ChromaDB.
[EvaluationAgent] Evaluating 60 transcript pairs...
[EvaluationAgent] Results: {'clean': 30, 'borderline': 22, 'failed': 1}. Borderline cases: 22

[HumanReview] 22 borderline case(s) require review:
  ID       : pair_009  (NICU)
  Reference: NICU team performed morning rounds...
  WER: 22.22%  CER: 5.36%
  ...

[main] Graph paused for human review.
       (In CLI mode, auto-approving all borderline cases for demo.)

[ReportAgent] Run run_20261013_212415 complete.
  Total: 60 | Clean: 30 | Borderline: 22 | Failed: 1
  Avg WER: 16.82% | Avg CER: 3.69% | Exact Match: 17/60
  Report saved → output/run_20261013_212415.json
```

---

## Step 5 — Run the tests

```bash
pytest tests/ -v
```

Expected: **23 passed** in ~12 seconds.

```
tests/test_agents.py::test_data_prep_removes_duplicates   PASSED
tests/test_agents.py::test_data_prep_keeps_unique_pairs   PASSED
tests/test_agents.py::test_data_prep_loads_kb_documents   PASSED
tests/test_metrics.py::test_wer_perfect_match             PASSED
tests/test_metrics.py::test_wer_complete_mismatch         PASSED
...
tests/test_rag.py::test_semantic_search_relevance         PASSED
tests/test_rag.py::test_semantic_search_category_filter   PASSED
23 passed in 11.92s
```

---

## Output files

Every pipeline run writes two files:

| File | Contents |
|---|---|
| `output/run_YYYYMMDD_HHMMSS.json` | Full report: summary metrics + per-pair results |
| `output/runs_history.json` | Append-only log used for regression detection |

---

## Bring your own data

Create a `.json` file matching this schema:

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

Or use the **Dataset Builder** in the UI to build this file interactively.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found: path/to/your/data.json` | That is the placeholder from the docs. Use `python main.py` or point `--data` at a real file. |
| `OPENAI_API_KEY not set` | Check `.env` exists and contains a valid key. |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt`. |
| Models download slowly | One-time only. They cache under `~/.cache/huggingface/`. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502`. |
| `chromadb.ClientAPI` error | Upgrade chromadb: `pip install --upgrade chromadb`. |
