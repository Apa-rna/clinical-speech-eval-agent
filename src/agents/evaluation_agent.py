from src.graph.state import EvalState, EvalResult
from src.evaluation.metrics import calculate_wer, calculate_cer, exact_match, classify_result
from src.rag.embeddings import semantic_search
from src.rag.reranker import rerank


def evaluation_agent(state: EvalState) -> dict:
    clean_pairs = state["clean_pairs"]
    eval_results: list[EvalResult] = []
    borderline_ids: list[str] = []

    print(f"[EvaluationAgent] Evaluating {len(clean_pairs)} transcript pairs...")

    for pair in clean_pairs:
        wer = calculate_wer(pair["reference"], pair["recognized"])
        cer = calculate_cer(pair["reference"], pair["recognized"])
        em = exact_match(pair["reference"], pair["recognized"])
        classification = classify_result(wer)

        # Retrieve clinical context for non-clean results
        clinical_context = ""
        if classification != "clean":
            query = f"{pair['category']} {pair['recognized']}"
            hits = semantic_search(query, k=3, category_filter=pair["category"])
            hits = rerank(query, hits, top_k=2)
            clinical_context = " | ".join(h["title"] for h in hits) if hits else ""

        result: EvalResult = {
            "id": pair["id"],
            "audio_file": pair["audio_file"],
            "category": pair["category"],
            "reference": pair["reference"],
            "recognized": pair["recognized"],
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "exact_match": em,
            "classification": classification,
            "clinical_context": clinical_context,
            "human_decision": None,
        }
        eval_results.append(result)

        if classification == "borderline":
            borderline_ids.append(pair["id"])

    by_class = {}
    for r in eval_results:
        by_class.setdefault(r["classification"], 0)
        by_class[r["classification"]] += 1
    print(f"[EvaluationAgent] Results: {by_class}. Borderline cases: {len(borderline_ids)}")

    return {"eval_results": eval_results, "borderline_ids": borderline_ids}
