from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, hits: list[dict], top_k: int = 2) -> list[dict]:
    if not hits:
        return hits
    model = _get_model()
    pairs = [[query, h["content"]] for h in hits]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked[:top_k]]
