import json
import tempfile
from pathlib import Path

from src.agents.data_prep_agent import data_prep_agent


def _make_dataset(pairs: list[dict]) -> str:
    data = {
        "transcript_pairs": pairs,
        "knowledge_base": [],
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def test_data_prep_removes_duplicates():
    pair = {
        "id": "pair_001",
        "audio_file": "audio_001.wav",
        "category": "NICU",
        "reference": "Patient in NICU",
        "recognized": "Patient in NICO",
        "expected_wer_range": "borderline",
    }
    duplicate = pair.copy()
    duplicate["id"] = "pair_002"

    path = _make_dataset([pair, duplicate])
    result = data_prep_agent({"dataset_path": path})

    assert len(result["raw_pairs"]) == 2
    assert len(result["clean_pairs"]) == 1


def test_data_prep_keeps_unique_pairs():
    pairs = [
        {
            "id": f"pair_{i:03d}",
            "audio_file": f"audio_{i}.wav",
            "category": "ICU",
            "reference": f"Reference text {i}",
            "recognized": f"Recognized text {i}",
            "expected_wer_range": "clean",
        }
        for i in range(5)
    ]
    path = _make_dataset(pairs)
    result = data_prep_agent({"dataset_path": path})
    assert len(result["clean_pairs"]) == 5


def test_data_prep_loads_kb_documents():
    data = {
        "transcript_pairs": [],
        "knowledge_base": [
            {"id": "kb_001", "category": "NICU", "title": "NICU", "content": "test"}
        ],
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()

    result = data_prep_agent({"dataset_path": tmp.name})
    assert len(result["kb_documents"]) == 1
