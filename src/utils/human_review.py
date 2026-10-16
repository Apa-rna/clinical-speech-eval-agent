from langgraph.types import interrupt

from src.graph.state import EvalState, EvalResult


def human_review_node(state: EvalState) -> dict:
    eval_results: list[EvalResult] = state["eval_results"]
    borderline_ids: set[str] = set(state["borderline_ids"])
    borderline_cases = [r for r in eval_results if r["id"] in borderline_ids]

    if not borderline_cases:
        return {"approved_results": eval_results}

    print(f"\n[HumanReview] {len(borderline_cases)} borderline case(s) require review:")
    print("-" * 70)
    for case in borderline_cases:
        print(f"  ID       : {case['id']}  ({case['category']})")
        print(f"  Reference: {case['reference']}")
        print(f"  Recognized: {case['recognized']}")
        print(f"  WER: {case['wer']:.2%}  CER: {case['cer']:.2%}")
        print(f"  Context  : {case['clinical_context'] or 'n/a'}")
        print()

    # Pause the graph and hand control to the human
    decisions: dict[str, str] = interrupt({
        "message": (
            "Review the borderline cases above. For each ID enter one of:\n"
            "  approve <id>         — accept the recognition as-is\n"
            "  reject <id>          — mark as failed\n"
            "  modify <id> <text>   — replace recognized text\n"
            "Type 'done' when finished."
        ),
        "borderline_ids": list(borderline_ids),
    })

    # Apply decisions
    approved: list[EvalResult] = []
    decision_map: dict[str, str] = decisions if isinstance(decisions, dict) else {}

    for result in eval_results:
        if result["id"] not in borderline_ids:
            approved.append(result)
            continue

        action = decision_map.get(result["id"], "approve")
        updated = dict(result)
        if action == "reject":
            updated["classification"] = "failed"
            updated["human_decision"] = "rejected"
        elif action.startswith("modify:"):
            updated["recognized"] = action.split("modify:", 1)[1].strip()
            updated["human_decision"] = "modified"
        else:
            updated["human_decision"] = "approved"
        approved.append(updated)

    print(f"[HumanReview] Decisions applied. Proceeding with {len(approved)} results.")
    return {"approved_results": approved}
