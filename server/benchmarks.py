"""
Serves the evaluation results (eval/metrics.py's output) to the dashboard, so the judged
numbers -- compound detection vs single-sensor baseline, FNR at matched FPR, zone
localization, lead time -- are presented in the product instead of buried in a JSON file.
The metrics are computed offline by `python -m eval.metrics` and committed; this endpoint
just reads that artifact.

/benchmarks/generalization serves a second, separate result: the held-out-zone test
(eval/permit_fusion_benchmark/held_out_zone.py) that answers "is the graph structurally
necessary" rather than "does the model detect well" -- the top table's benchmark shows
the GNN matches classical baselines in-distribution, so the case for the graph rests on
this held-out-zone result, not on the top table's headline numbers. See
docs/external-validation-findings.md Sec 4 for the full writeup.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "eval" / "results" / "metrics.json"
HELDOUT_DIR = REPO_ROOT / "eval" / "permit_fusion_benchmark"
HELDOUT_FILES = {
    "feed_zone": HELDOUT_DIR / "results_heldout_feed_zone.json",
    "condenser_zone": HELDOUT_DIR / "results_heldout_condenser_zone.json",
}

router = APIRouter(tags=["benchmarks"])


@router.get("/benchmarks")
def get_benchmarks():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404,
                            detail="No evaluation results found -- run `python -m eval.metrics` first")
    return json.loads(METRICS_PATH.read_text())


@router.get("/benchmarks/generalization")
def get_generalization_benchmarks():
    runs = {name: json.loads(path.read_text()) for name, path in HELDOUT_FILES.items() if path.exists()}
    if not runs:
        raise HTTPException(
            status_code=404,
            detail="No held-out-zone results found -- run "
                    "`python -m eval.permit_fusion_benchmark.held_out_zone <scenario_id>` first",
        )
    return {"runs": runs}
