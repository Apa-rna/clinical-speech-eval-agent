import json
from pathlib import Path

_STORE_PATH = Path("output/runs_history.json")


def _load_store() -> list[dict]:
    if _STORE_PATH.exists():
        with open(_STORE_PATH) as f:
            return json.load(f)
    return []


def _save_store(runs: list[dict]) -> None:
    _STORE_PATH.parent.mkdir(exist_ok=True)
    with open(_STORE_PATH, "w") as f:
        json.dump(runs, f, indent=2)


def load_latest_run() -> dict | None:
    runs = _load_store()
    return runs[-1] if runs else None


def save_run(run_id: str, report: dict) -> None:
    runs = _load_store()
    runs.append({"run_id": run_id, **report})
    _save_store(runs)
