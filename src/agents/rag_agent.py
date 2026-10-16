from src.graph.state import EvalState
from src.rag.embeddings import index_documents


def rag_agent(state: EvalState) -> dict:
    kb_documents = state["kb_documents"]
    added = index_documents(kb_documents)
    print(f"[RAGAgent] Indexed {added} new KB document(s) into ChromaDB "
          f"({len(kb_documents)} total in collection).")
    return {}
