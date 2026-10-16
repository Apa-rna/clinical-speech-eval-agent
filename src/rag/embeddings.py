from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions


_COLLECTION_NAME = "clinical_kb"
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.Client()
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection
    client = _get_client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    _collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def index_documents(documents: list[dict]) -> int:
    collection = get_collection()
    existing = set(collection.get()["ids"])
    to_add = [d for d in documents if d["id"] not in existing]
    if not to_add:
        return 0
    collection.upsert(
        ids=[d["id"] for d in to_add],
        documents=[d["content"] for d in to_add],
        metadatas=[{"category": d["category"], "title": d["title"]} for d in to_add],
    )
    return len(to_add)


def semantic_search(query: str, k: int = 3, category_filter: str | None = None) -> list[dict]:
    collection = get_collection()
    where = {"category": category_filter} if category_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "content": doc,
            "category": meta.get("category"),
            "title": meta.get("title"),
            "score": 1 - dist,
        })
    return hits
