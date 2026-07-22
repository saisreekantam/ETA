"""
Operator chat: natural-language Q&A over the plant's CURRENT state and the regulatory
corpus, answered by the same local LLM (Ollama) and pgvector retrieval the incident
reports use -- no cloud API, plant data never leaves the network.

Each question gets three kinds of grounding, all assembled server-side:
  1. RAG: top-5 regulatory chunks for the question (rag/retriever.py), cited back in the
     response the same way incident reports cite them -- the model is told to quote only
     these, so it can't invent section numbers.
  2. Plant state: the latest persisted per-zone risk scores, contributing sensors,
     unacknowledged alerts, and the most recent incident report. This is what lets an
     operator ask "why is reactor_zone flagged?" and get an answer grounded in the actual
     numbers the GNN just produced, not a generic essay.
  3. The app guide (docs/app-guide.md), so "how do I add a sensor?" gets the real
     workflow, not a hallucinated one.

COPILOT: the model can also drive the dashboard. It gets a small tool set (navigate,
run_scenario, start_replay) via Ollama's /api/chat tool-calling; chosen tools come back
to the frontend as `actions`, which ChatPanel executes -- "show me the reactor replay"
both answers AND switches the view. Falls back to plain /api/generate answering if the
model/endpoint doesn't support tools.

MEMORY: a short rolling history (last MAX_HISTORY_TURNS turns) rides along in the prompt
for follow-ups ("what about the stripper?"). The endpoint stays stateless -- the thread
lives client-side in sessionStorage, scoped per facility (see ChatPanel.jsx), so it
survives a reload but dies with the tab and never crosses between plants.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.nodes.orchestrator_node import OLLAMA_MODEL, _call_llm
from db.models import Alert, Facility, IncidentReport, Run, Zone, ZoneRiskScore
from db.session import get_db
from db.settings import settings
from rag.retriever import retrieve

router = APIRouter(tags=["chat"])

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_FACILITY_NAME = "Demo Steel & Chemical Plant"
MAX_HISTORY_TURNS = 6

_APP_GUIDE_PATH = REPO_ROOT / "docs" / "app-guide.md"
APP_GUIDE = _APP_GUIDE_PATH.read_text() if _APP_GUIDE_PATH.exists() else ""

# UI tools the model may call; ChatPanel.jsx executes these client-side. Kept few and
# coarse on purpose: navigation is where a copilot helps, not a substitute for the UI.
CHAT_TOOLS = [
    {"type": "function", "function": {
        "name": "navigate",
        "description": "Switch the dashboard to a page. Use when the operator asks to see/open something.",
        "parameters": {"type": "object", "properties": {
            "page": {"type": "string", "enum": ["single", "replay", "live", "devices", "evaluation", "about"],
                      "description": "single=scenario runs + plant risk map"}},
            "required": ["page"]},
    }},
    {"type": "function", "function": {
        "name": "run_scenario",
        "description": "Run a benchmark scenario through the detection pipeline and show the result on the plant map. Pick a run_id from the scenario list in context.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}}, "required": ["run_id"]},
    }},
    {"type": "function", "function": {
        "name": "start_replay",
        "description": "Open the time-replay view for a run so the operator can scrub its risk timeline.",
        "parameters": {"type": "object", "properties": {
            "run_id": {"type": "string"}}, "required": ["run_id"]},
    }},
]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = []
    facility_id: str | None = None


def _resolve_facility_id(db: Session, facility_id: str | None) -> uuid.UUID:
    if facility_id:
        return uuid.UUID(facility_id)
    facility = db.query(Facility).filter_by(name=DEMO_FACILITY_NAME).first()
    if facility is None:
        raise HTTPException(status_code=500, detail="No facility seeded -- run `python -m db.seed` first")
    return facility.id


def _plant_state_block(db: Session, fid: uuid.UUID) -> str:
    """Compact text snapshot of the facility and its latest persisted state. Latest
    score per zone (zones without any persisted score are omitted), open alerts,
    latest report."""
    facility = db.query(Facility).get(fid)
    zones = db.query(Zone).filter_by(facility_id=fid).all()
    key_by_zone_id = {z.id: z.key for z in zones}
    header = [f"Facility: {facility.name}"
              + (f" ({facility.industry_type.replace('_', ' ')})" if facility.industry_type else "")
              + (f", {facility.location}" if facility.location else ""),
              "Monitored zones: " + ", ".join(f"{z.label} [{z.key}]" for z in zones)]

    latest_by_zone: dict[str, ZoneRiskScore] = {}
    recent = (db.query(ZoneRiskScore).filter_by(facility_id=fid)
              .order_by(ZoneRiskScore.computed_at.desc()).limit(200).all())
    for s in recent:
        zone_key = key_by_zone_id.get(s.zone_id)
        if zone_key and zone_key not in latest_by_zone:
            latest_by_zone[zone_key] = s

    lines = header
    if latest_by_zone:
        lines.append("Latest per-zone compound-risk scores (GNN, 0-1) with top contributing sensors:")
        for zone_key, s in sorted(latest_by_zone.items(), key=lambda kv: -kv[1].compound_risk_score):
            sensors = ", ".join(s.contributing_sensors[:3]) if s.contributing_sensors else "n/a"
            lines.append(f"- {zone_key}: risk={s.compound_risk_score:.2f} "
                         f"(baseline {s.baseline_risk_score:.2f}), key sensors: {sensors}, "
                         f"computed {s.computed_at.isoformat(timespec='minutes')}")
    else:
        lines.append("No risk scores computed yet (no scenario has been run).")

    open_alerts = (db.query(Alert).filter_by(facility_id=fid, acknowledged_by=None)
                   .order_by(Alert.created_at.desc()).limit(5).all())
    if open_alerts:
        lines.append("Unacknowledged alerts:")
        lines.extend(f"- [{a.level}] {a.message}" for a in open_alerts)

    report = (db.query(IncidentReport).filter_by(facility_id=fid)
              .order_by(IncidentReport.created_at.desc()).first())
    if report:
        lines.append(f"Most recent incident report ({report.escalation_level}, "
                     f"zone {report.zone.key}): {report.report_text[:400]}")
    return "\n".join(lines)


def _scenario_block(db: Session, fid: uuid.UUID) -> str:
    """Compact scenario/run list so the model can pick real run_ids for its tools."""
    runs = (db.query(Run).filter_by(facility_id=fid)
            .filter(Run.condition.in_(["compound", "normal"])).order_by(Run.scenario_id).all())
    by_scenario: dict[str, dict[str, str]] = {}
    for r in runs:
        bucket = by_scenario.setdefault(r.scenario_id, {})
        if r.condition not in bucket:
            bucket[r.condition] = r.external_run_id
    if not by_scenario:
        return "No benchmark scenarios on this facility (they live on the demo plant)."
    return "\n".join(f"- {sid}: compound run_id={ids.get('compound', 'n/a')}, "
                     f"normal run_id={ids.get('normal', 'n/a')}"
                     for sid, ids in by_scenario.items())


def _ollama_chat(messages: list[dict], model: str, base_url: str) -> dict | None:
    """One /api/chat round with tools. Returns the response message dict, or None if
    the endpoint/model can't do tool chat (caller falls back to plain generation)."""
    try:
        resp = requests.post(f"{base_url.rstrip('/')}/api/chat",
                             json={"model": model, "messages": messages,
                                   "tools": CHAT_TOOLS, "stream": False}, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]
    except (requests.exceptions.RequestException, KeyError):
        return None


@router.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    fid = _resolve_facility_id(db, req.facility_id)

    citations = retrieve(req.question, k=5)
    citation_block = "\n".join(f"- {c.citation}: \"{c.text[:300].strip()}\"" for c in citations)
    plant_block = _plant_state_block(db, fid)
    scenario_block = _scenario_block(db, fid)

    system = f"""You are the safety-operations copilot for an Indian industrial plant. Always answer in
English. Be concise and concrete (under 180 words). Ground every claim in the plant
state, the app guide, or the regulatory text below. Cite regulatory text ONLY from the listed excerpts, by their
citation labels, verbatim -- never invent section numbers or cite anything not listed.
If nothing below answers the question, say so plainly.

You can also DRIVE the dashboard with your tools (navigate / run_scenario /
start_replay). Use a tool when the operator asks to see, open, run, or replay something
-- and still answer in words. Never invent run_ids: use only ids from the scenario list.

Current plant state:
{plant_block}

Benchmark scenarios available:
{scenario_block}

App guide (how the operator does things):
{APP_GUIDE}

Retrieved regulatory text (cite these, and only these):
{citation_block}"""

    messages = ([{"role": "system", "content": system}]
                + [{"role": m.role, "content": m.content} for m in req.history[-MAX_HISTORY_TURNS:]]
                + [{"role": "user", "content": req.question}])

    model = settings.chat_model or OLLAMA_MODEL
    base_url = settings.chat_ollama_url or settings.ollama_url

    actions = []
    answer = None
    msg = _ollama_chat(messages, model, base_url)
    if msg is not None:
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name") in {"navigate", "run_scenario", "start_replay"}:
                actions.append({"tool": fn["name"], "args": fn.get("arguments") or {}})
        answer = (msg.get("content") or "").strip()
        if actions and not answer:
            # Model chose tools but said nothing -- one more round for the words.
            # Deliberately NOT using the "tool" role here: qwen's chat template mangles
            # bare tool-result turns through Ollama (answers came back in Thai);
            # restating the invocation as plain text is template-safe.
            done = ", ".join(f"{a['tool']}({a['args']})" for a in actions)
            followup = (messages
                        + [{"role": "assistant", "content": f"[invoked: {done}]"},
                           {"role": "user", "content": "That action is now running in the dashboard. "
                            "In English, briefly confirm what you did and answer my original question."}])
            final = _ollama_chat(followup, model, base_url)
            answer = (final.get("content") or "").strip() if final else ""

    if not answer:
        # tool-chat unavailable (or empty twice): plain generation, no actions
        prompt = system + f"\n\nOperator's question: {req.question}\n\nAnswer:"
        chat_url = (f"{settings.chat_ollama_url.rstrip('/')}/api/generate"
                    if settings.chat_ollama_url else None)
        answer = _call_llm(prompt, url=chat_url, model=settings.chat_model or None)

    return {
        "answer": answer,
        "model": model,
        "actions": actions,
        "citations": [{"source": c.citation, "text": c.text, "score": round(c.score, 3)}
                      for c in citations],
    }
