from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import EvalState
from src.agents.data_prep_agent import data_prep_agent
from src.agents.rag_agent import rag_agent
from src.agents.evaluation_agent import evaluation_agent
from src.agents.report_agent import report_agent
from src.utils.human_review import human_review_node


def _has_borderline(state: EvalState) -> str:
    return "human_review" if state.get("borderline_ids") else "report"


def build_graph() -> StateGraph:
    builder = StateGraph(EvalState)

    builder.add_node("data_prep", data_prep_agent)
    builder.add_node("rag_index", rag_agent)
    builder.add_node("evaluate", evaluation_agent)
    builder.add_node("human_review", human_review_node)
    builder.add_node("report", report_agent)

    builder.set_entry_point("data_prep")
    builder.add_edge("data_prep", "rag_index")
    builder.add_edge("rag_index", "evaluate")
    builder.add_conditional_edges("evaluate", _has_borderline, {
        "human_review": "human_review",
        "report": "report",
    })
    builder.add_edge("human_review", "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=MemorySaver())
