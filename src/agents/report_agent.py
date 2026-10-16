import json
import datetime
from pathlib import Path

from src.graph.state import EvalState, EvalResult, RunComparison
from src.memory.run_store import load_latest_run, save_run


def report_agent(state: EvalState) -> dict:
    results: list[EvalResult] = state.get("approved_results") or state.get("eval_results", [])
    run_comparison: RunComparison | None = state.get("run_comparison")

    # Aggregate metrics
    total = len(results)
    clean = sum(1 for r in results if r["classification"] == "clean")
    borderline = sum(1 for r in results if r["classification"] == "borderline")
    failed = sum(1 for r in results if r["classification"] == "failed")
    avg_wer = round(sum(r["wer"] for r in results) / total, 4) if total else 0.0
    avg_cer = round(sum(r["cer"] for r in results) / total, 4) if total else 0.0
    exact_matches = sum(1 for r in results if r["exact_match"])

    # Compare with previous run
    prev_run = load_latest_run()
    regression_ids: list[str] = []
    improvement_ids: list[str] = []

    if prev_run:
        prev_map = {r["id"]: r for r in prev_run.get("results", [])}
        for r in results:
            prev = prev_map.get(r["id"])
            if prev:
                if r["wer"] > prev["wer"] + 0.05:
                    regression_ids.append(r["id"])
                elif r["wer"] < prev["wer"] - 0.05:
                    improvement_ids.append(r["id"])
        prev_avg_wer = prev_run.get("summary", {}).get("avg_wer", avg_wer)
        run_comparison = RunComparison(
            previous_run_id=prev_run.get("run_id"),
            avg_wer_delta=round(avg_wer - prev_avg_wer, 4),
            regressions=regression_ids,
            improvements=improvement_ids,
        )

    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    report = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_pairs": total,
            "clean": clean,
            "borderline": borderline,
            "failed": failed,
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "exact_matches": exact_matches,
            "exact_match_rate": round(exact_matches / total, 4) if total else 0.0,
        },
        "run_comparison": run_comparison,
        "results": results,
    }

    # Save run to memory
    save_run(run_id, report)

    # Write output file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[ReportAgent] Run {run_id} complete.")
    print(f"  Total: {total} | Clean: {clean} | Borderline: {borderline} | Failed: {failed}")
    print(f"  Avg WER: {avg_wer:.2%} | Avg CER: {avg_cer:.2%} | Exact Match: {exact_matches}/{total}")
    if run_comparison and run_comparison.get("previous_run_id"):
        delta = run_comparison["avg_wer_delta"]
        sign = "+" if delta > 0 else ""
        print(f"  vs {run_comparison['previous_run_id']}: WER delta {sign}{delta:.2%} | "
              f"Regressions: {len(regression_ids)} | Improvements: {len(improvement_ids)}")
    print(f"  Report saved → {out_path}")

    return {"final_report": report, "run_comparison": run_comparison}
