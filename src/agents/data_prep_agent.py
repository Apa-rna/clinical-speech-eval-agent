import hashlib
import json
from pathlib import Path

from src.graph.state import EvalState, TranscriptPair


def data_prep_agent(state: EvalState) -> dict:
    dataset_path = state["dataset_path"]
    with open(dataset_path) as f:
        data = json.load(f)

    raw_pairs: list[TranscriptPair] = data["transcript_pairs"]
    kb_documents: list[dict] = data["knowledge_base"]

    # Deduplicate by hash of (reference, recognized)
    seen: set[str] = set()
    clean_pairs: list[TranscriptPair] = []
    for pair in raw_pairs:
        key = hashlib.md5(
            f"{pair['reference']}|{pair['recognized']}".encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            clean_pairs.append(pair)

    duplicates_removed = len(raw_pairs) - len(clean_pairs)
    print(f"[DataPrepAgent] Loaded {len(raw_pairs)} pairs → {len(clean_pairs)} after dedup "
          f"({duplicates_removed} duplicate(s) removed). KB: {len(kb_documents)} docs.")

    return {
        "raw_pairs": raw_pairs,
        "clean_pairs": clean_pairs,
        "kb_documents": kb_documents,
    }
