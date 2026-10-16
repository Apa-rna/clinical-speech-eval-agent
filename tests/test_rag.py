import pytest
from src.rag.embeddings import index_documents, semantic_search, get_collection


SAMPLE_DOCS = [
    {
        "id": "test_kb_001",
        "category": "NICU",
        "title": "NICU Overview",
        "content": "NICU stands for Neonatal Intensive Care Unit for premature newborns.",
    },
    {
        "id": "test_kb_002",
        "category": "ICU",
        "title": "ICU Overview",
        "content": "ICU stands for Intensive Care Unit providing critical adult care.",
    },
    {
        "id": "test_kb_003",
        "category": "ED",
        "title": "ED Overview",
        "content": "ED stands for Emergency Department handling acute emergency cases.",
    },
]


@pytest.fixture(autouse=True)
def index_sample_docs():
    index_documents(SAMPLE_DOCS)


def test_index_documents_no_duplicates():
    added = index_documents(SAMPLE_DOCS)
    assert added == 0  # already indexed, should add 0


def test_semantic_search_returns_results():
    results = semantic_search("premature newborn care unit", k=2)
    assert len(results) > 0


def test_semantic_search_relevance():
    results = semantic_search("neonatal premature baby intensive care", k=3)
    titles = [r["title"] for r in results]
    assert "NICU Overview" in titles


def test_semantic_search_category_filter():
    results = semantic_search("intensive care", k=3, category_filter="ICU")
    assert all(r["category"] == "ICU" for r in results)


def test_semantic_search_score_range():
    results = semantic_search("emergency acute department", k=2)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0
