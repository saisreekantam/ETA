"""
Wires the three nodes into a LangGraph StateGraph with SQLite checkpointing -- the
checkpoint DB is the audit trail: every run's full state at every node is replayable,
which is what lets the demo answer "why did the system flag this at sample 62" by
inspecting the saved state rather than trusting a log line.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agents.nodes.compound_risk_node import compound_risk_node
from agents.nodes.orchestrator_node import orchestrator_node
from agents.nodes.permit_correlation_node import permit_correlation_node
from agents.state import PipelineState

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DB = REPO_ROOT / "agents" / "checkpoints" / "audit_trail.sqlite"


def build_pipeline():
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    graph = StateGraph(PipelineState)

    graph.add_node("compound_risk", compound_risk_node)
    graph.add_node("permit_correlation", permit_correlation_node)
    graph.add_node("orchestrator", orchestrator_node)

    graph.set_entry_point("compound_risk")
    graph.add_edge("compound_risk", "permit_correlation")
    graph.add_edge("permit_correlation", "orchestrator")
    graph.add_edge("orchestrator", END)

    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)
