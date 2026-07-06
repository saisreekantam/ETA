"""
Continuous live plant mode: a server-paced SSE stream that replays a run's sensor trace
through the same GNN scoring path as /replay, one frame per tick, looping -- so the plant
map moves on its own like a real control-room wallboard instead of waiting for a click.

Reuses compute_replay_trace (identical scoring math to Time Replay); the trace for a run
is computed once and cached, so the stream itself is just a timer over precomputed
frames. The frontend consumes this with a plain EventSource.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from scripts.replay import compute_replay_trace

router = APIRouter(tags=["live"])

RISK_ALERT_THRESHOLD = 0.9


@lru_cache(maxsize=8)
def _cached_trace(run_id: str, step: int) -> dict:
    return compute_replay_trace(run_id, step=step)


@router.get("/live/risk-stream")
def live_risk_stream(run_id: str, interval: float = 1.2, step: int = 4):
    try:
        trace = _cached_trace(run_id, step)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run_id {run_id} not found")

    cutoffs = trace["cutoffs"]
    zones = list(trace["trace"].keys())
    onset = trace["ground_truth_onset_sample"]

    def frames():
        while True:  # loop forever -- the client closes the stream when it leaves
            alerted: set[str] = set()
            for i, cutoff in enumerate(cutoffs):
                zone_scores = {}
                new_alerts = []
                for z in zones:
                    gnn = trace["trace"][z]["gnn"][i]
                    zone_scores[z] = {
                        "gnn": gnn,
                        "baseline_alert": trace["trace"][z]["baseline_alert"][i],
                    }
                    if gnn >= RISK_ALERT_THRESHOLD and z not in alerted:
                        alerted.add(z)
                        new_alerts.append(z)
                payload = {
                    "t": cutoff,
                    "t_index": i,
                    "n_frames": len(cutoffs),
                    "true_zone": trace["true_zone"],
                    "onset": onset,
                    "onset_reached": onset is not None and cutoff >= onset,
                    "zones": zone_scores,
                    "new_alerts": new_alerts,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(interval)

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
