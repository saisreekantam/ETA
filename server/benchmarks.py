"""
Serves the evaluation results (eval/metrics.py's output) to the dashboard, so the judged
numbers -- compound detection vs single-sensor baseline, FNR at matched FPR, zone
localization, lead time -- are presented in the product instead of buried in a JSON file.
The metrics are computed offline by `python -m eval.metrics` and committed; this endpoint
just reads that artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "eval" / "results" / "metrics.json"

router = APIRouter(tags=["benchmarks"])


@router.get("/benchmarks")
def get_benchmarks():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404,
                            detail="No evaluation results found -- run `python -m eval.metrics` first")
    return json.loads(METRICS_PATH.read_text())
