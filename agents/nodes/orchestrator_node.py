"""
Emergency Response Orchestrator node. Synthesizes the compound-risk score and permit
violations, retrieves real regulatory text via rag/retriever.py, and asks a local LLM
(Ollama, no cloud API call -- the data-sovereignty pitch: SCADA/permit data never leaves
the plant network) to draft an incident report that cites the retrieved text verbatim.

The LLM is explicitly instructed to quote retrieved text rather than invent citations --
this is enforced structurally (citations are appended programmatically, not left to the
model to recall from its own memory) so a generated report can never cite a regulation
section that wasn't actually retrieved.
"""
from __future__ import annotations

import requests

from agents.state import PipelineState, RetrievedCitation
from db.settings import settings
from rag.retriever import retrieve

# Base from settings (env OLLAMA_URL) so this works both locally (localhost) and in
# Docker (http://ollama:11434) without a code change.
OLLAMA_URL = f"{settings.ollama_url.rstrip('/')}/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
# llama3.1:8b doesn't fit fully in 8 GB VRAM alongside the torch CUDA context;
# offload some layers to CPU so the alloc succeeds (0/None = let Ollama decide
# on cards with enough VRAM). Tune up if your GPU is larger.
OLLAMA_NUM_GPU = 20

ESCALATION_BANDS = [(0.9, "emergency"), (0.7, "alert"), (0.5, "monitor"), (0.0, "none")]


def _escalation_for(max_score: float) -> str:
    for threshold, label in ESCALATION_BANDS:
        if max_score >= threshold:
            return label
    return "none"


def _call_llm(prompt: str) -> str:
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        if OLLAMA_NUM_GPU:
            payload["options"] = {"num_gpu": OLLAMA_NUM_GPU}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        return f"[LLM unavailable ({e}) -- falling back to template report below]"


def orchestrator_node(state: PipelineState) -> dict:
    if not state["permit_violations"]:
        audit_entry = "orchestrator_node: no permit violations, no report generated"
        return {"escalation_level": "none", "audit_log": state["audit_log"] + [audit_entry]}

    severity_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    top_violation = max(state["permit_violations"], key=lambda v: severity_rank[v["severity"]])
    permit_type = next((p["permit_type"] for p in state["permits"] if p["permit_id"] == top_violation["permit_id"]), "")
    query = f"{permit_type} permit {top_violation['zone']} precautions hazard"
    citations = retrieve(query, k=3)

    citation_block = "\n".join(f"- {c.citation}: \"{c.text[:300].strip()}\"" for c in citations)
    prompt = f"""You are a process safety analyst at an Indian industrial plant. Draft a brief
(under 200 words) preliminary incident analysis for the following compound-risk detection.
Cite ONLY the regulatory text given below verbatim with its citation label -- do not invent
section numbers or quote any regulation not listed here.

Detection: {top_violation['reason']}
Zone: {top_violation['zone']}
Severity: {top_violation['severity']}

Retrieved regulatory text (cite these, and only these):
{citation_block}

Write the incident analysis now."""

    report = _call_llm(prompt)
    max_score = max(r["compound_risk_score"] for r in state["zone_risk_scores"])
    escalation = _escalation_for(max_score)

    retrieved = [RetrievedCitation(source=c.citation, text=c.text, score=c.score) for c in citations]
    audit_entry = (f"orchestrator_node: escalation={escalation}, "
                    f"cited {len(retrieved)} source(s): {[c.citation for c in citations]}")

    return {
        "retrieved_citations": retrieved,
        "incident_report": report,
        "escalation_level": escalation,
        "audit_log": state["audit_log"] + [audit_entry],
    }
