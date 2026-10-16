"""
Clinical Speech Evaluation Agent — Entry Point

Usage:
    python main.py                         # run with default synthetic dataset
    python main.py --data path/to/data.json
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command
from src.graph.workflow import build_graph


def main():
    parser = argparse.ArgumentParser(description="Clinical Speech Evaluation Agent")
    parser.add_argument(
        "--data",
        default="data/synthetic_transcripts.json",
        help="Path to transcript dataset JSON",
    )
    parser.add_argument(
        "--thread-id",
        default="session-1",
        help="LangGraph thread ID (change to start a fresh run)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Run first: python data/generate_synthetic_data.py")
        sys.exit(1)

    print("=" * 60)
    print("  Clinical Speech Evaluation Agent")
    print("=" * 60)

    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread_id}}
    initial_state = {"dataset_path": str(data_path)}

    # Stream events so progress is visible
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        pass

    # Handle HITL interrupt if graph paused
    snapshot = graph.get_state(config)
    if snapshot.next:
        print("\n[main] Graph paused for human review.")
        print("       (In CLI mode, auto-approving all borderline cases for demo.)")
        print("       Pass decisions dict to graph.invoke(..., input={...}) to override.\n")

        # Auto-approve all borderline for non-interactive demo
        borderline_ids = snapshot.values.get("borderline_ids", [])
        decisions = {bid: "approve" for bid in borderline_ids}

        for event in graph.stream(Command(resume=decisions), config=config, stream_mode="values"):
            pass

    final = graph.get_state(config)
    report = final.values.get("final_report")
    if report:
        print(f"\nDone. Report: output/{report['run_id']}.json")
    else:
        print("\nNo report generated — check for errors above.")


if __name__ == "__main__":
    main()
