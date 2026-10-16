from typing import TypedDict, Optional, Annotated
import operator


class TranscriptPair(TypedDict):
    id: str
    audio_file: str
    category: str
    reference: str
    recognized: str
    expected_wer_range: str


class EvalResult(TypedDict):
    id: str
    audio_file: str
    category: str
    reference: str
    recognized: str
    wer: float
    cer: float
    exact_match: bool
    classification: str          # "clean" | "borderline" | "failed"
    clinical_context: str        # retrieved from RAG
    human_decision: Optional[str]  # "approved" | "rejected" | "modified"


class RunComparison(TypedDict):
    previous_run_id: Optional[str]
    avg_wer_delta: Optional[float]
    regressions: list[str]       # pair IDs that got worse
    improvements: list[str]      # pair IDs that got better


class EvalState(TypedDict):
    dataset_path: str
    raw_pairs: list[TranscriptPair]
    clean_pairs: list[TranscriptPair]
    kb_documents: list[dict]
    eval_results: list[EvalResult]
    borderline_ids: list[str]
    approved_results: list[EvalResult]
    run_comparison: Optional[RunComparison]
    final_report: Optional[dict]
    error: Optional[str]
