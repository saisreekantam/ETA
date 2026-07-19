# Industrial Safety Intelligence

**Compound-risk detection for industrial plants — catching dangerous combinations that no single sensor would flag alone.**

Built for ET AI Hackathon 2026, Problem 1: *AI-Powered Industrial Safety Intelligence for Zero-Harm Operations*.

Sensors, permits, and CCTV in most plants live in three different systems that never talk to each other. A gas sensor can be nominal, a confined-space permit can be active, and a camera can show five unhelmeted workers in that zone — all true at once, all individually unremarkable, and collectively the exact failure mode behind incidents like the Jan 2025 Visakhapatnam coke-oven explosion (8 dead) and the Jun 2025 Sigachi Industries dust explosion (46 dead). This platform fuses sensor telemetry, permit-to-work logs, worker presence, and live CCTV into one risk signal, and explains *why* in plain language with regulatory citations — not just a number on a dashboard.

## What it does

- **Compound-risk detection** — a heterogeneous Graph Attention Network (GATv2) scores every plant zone in real time by fusing sensor readings, active permits, and worker presence, catching combinations a per-sensor threshold alarm would miss entirely.
- **Permit correlation** — flags when a permit (e.g. confined-space, hot-work) is active in a zone the model has independently scored as high-risk, and surfaces the specific CCTV evidence (unhelmeted workers, unauthorized entry) backing that call.
- **RAG-grounded incident reports** — a local LLM (no cloud API — sensor/safety data never leaves the facility) drafts an incident report citing the actual regulatory text retrieved from a DGMS/OISD/Factory Act corpus, not a hallucinated reference.
- **Live CCTV hazard detection** — four fine-tuned RT-DETR models running on real video (uploaded clip or a live camera/RTSP feed, same code path either way): PPE compliance, unauthorized zone entry, fall/man-down, fire/smoke. Detections don't just render in the session view: they raise persisted, acknowledgeable alerts (+ outbound webhook) with per-event cooldowns, so a hazard seen at 30fps warns once, immediately, instead of flooding the inbox (`server/live_watch.py`).
- **Live IoT anomaly watch** — a background watcher evaluates every ingested sensor reading against that device's own trailing baseline (no per-device configuration beyond assigning a zone); deviations beyond 4σ raise the same acknowledgeable alerts. The operator chat sees unacknowledged alerts in its plant-state context, so "anything I should know?" surfaces live warnings too.
- **Full audit trail** — every agent decision is checkpointed (LangGraph + SQLite), and every run, violation, and incident is persisted to a multi-tenant Postgres backend.
- **Operator copilot** — a side-panel assistant (`POST /chat`, same local LLM + pgvector retrieval as the reports) that answers natural-language questions grounded in the live plant state ("why is reactor_zone flagged?"), the regulatory corpus, and the app's own user guide — and can drive the dashboard via tool-calling ("run the reactor scenario" actually runs it; "how do I add a sensor?" gets the real workflow). Citations attached to every answer.
- **IoT ingest security** — every registered sensor gets a per-device token; `/iot/readings` rejects unknown devices and bad tokens with a persisted `security` alert (cooldown-deduplicated), so a spoofing attempt is itself a detected, auditable event rather than silent data poisoning.
- **Explainable zone scores** — hovering a zone on the risk map shows a mini bar chart of the gradient-based sensor saliency behind its score ("XMEAS(7) at 100%, XMEAS(8) at 17%"), computed per-instrument by `models/gnn/attribution.py`, not a canned sensor list. The map's process pipes render the GATv2's actual `zone→zone` attention scaled by source-zone risk — risk visibly propagating downstream is the model's edge attention, not an animation gimmick.

## Results (real numbers, not cherry-picked)

Evaluated on a held-out, run-level test split (120 runs, 30 compound-positive) — see [`eval/results/metrics.json`](eval/results/metrics.json):

| Metric | GNN | Single-sensor baseline |
|---|---|---|
| F1 (compound-risk detection) | **100%** | 54.1% |
| Precision | **100%** | 37.0% |
| Zone-localization accuracy | **100%** | 23.3% |
| False-negative rate @ 5% FPR | **0%** | 90% |
| Median detection latency vs. ground-truth onset | **0 min (at onset)** | 0 min (at onset) |

(The previous mean-pooled model scored F1 95.2% / precision 90.9% and detected a median 18 min after onset; adding a GRU temporal encoder over the raw 30-sample window plus per-sensor/worker/plant graph nodes closed both gaps. Perfect scores on a 120-run synthetic test split say the benchmark is now saturated, not that the problem is solved.)

Live CCTV models, evaluated on their respective held-out test sets:

| Detector | Class | mAP50 |
|---|---|---|
| Fire/smoke (D-Fire, 4302 test images) | smoke | 0.82 |
| Fire/smoke (D-Fire, 4302 test images) | fire | 0.70 |
| Fall/man-down | down | 0.74 |
| Fall/man-down | crouching/bending | 0.76 |
| PPE compliance | head/helmet detection (used for violation logic) | — |

The single-sensor baseline is a genuinely naive per-sensor z-score/EWMA threshold trained on the same data — it's the strawman the GNN is meant to beat, not a weak strawman dressed up.

## Architecture

```
TEP process simulator + fault injection  ──┐
synthetic permits / worker-presence logs ──┼──> heterogeneous graph ──> GATv2 GNN
                                            │     (per-zone compound-risk score)
                              (naive single-sensor                │
                               threshold baseline,                 v
                               same data, for comparison)   LangGraph pipeline:
                                                              compound_risk_node
                                                           -> permit_correlation_node
                                                           -> orchestrator_node
                                                              (RAG retrieval over DGMS/
                                                               OISD/Factory Act corpus +
                                                               local LLM incident report,
                                                               SQLite checkpoint = audit trail)
                                                                       │
live CCTV (file or RTSP) ──> RT-DETR x4 (PPE / fall / fire-smoke /    │
                              zone-intrusion) ──> live event feed     │
                                                                       v
                                          FastAPI (Postgres + pgvector, API-key auth)
                                                            │
                                                            v
                                          React dashboard — zone risk map, time replay,
                                          live CCTV view, audit trail, incident reports
```

## Why these choices

- **TEP simulator, not a toy dataset** — the Tennessee Eastman Process gives 52 real process variables and a physically grounded substrate for injecting dual-fault scenarios with an actual mechanistic "compounding" story (e.g. reactor cooling-water temperature step + valve sticking — neither alone crosses a single-sensor alarm threshold, the combination genuinely impairs heat removal). See [`docs/scenario-definitions.md`](docs/scenario-definitions.md) for all five scenarios and the TEP-unit-to-zone mapping.
- **GATv2 over a heterogeneous graph, not a flat classifier** — sensor, sensor-cluster, permit, worker-presence, and plant nodes have genuinely different relationships to a zone; a hetero-GNN lets the model learn zone-flow propagation (a fault in one unit raising risk in the next) instead of treating every signal as interchangeable. Sensor clusters enter the graph through a GRU over the raw 30-sample window rather than mean-pooled summary stats, so a slow drift and a sudden step — identical in window mean — are distinguishable.
- **LangGraph, not a single LLM call** — explicit state graph with built-in checkpointing gives a free, inspectable audit trail, which is what the "regulatory compliance coverage" judging criterion actually rewards.
- **RT-DETR, not YOLO** — a transformer-based detector gives comparable inference speed pretrained, plus attention maps as a built-in explainability artifact for "why did the agent flag this frame."
- **Local LLM (Ollama), not a cloud API** — plants will not send live SCADA/safety data to an external API; this runs entirely on-prem.
- **Multi-tenant from the schema up** — every relational table anchors on `facility_id`, not bolted on later.

## Deployment

Three tiers, same containers throughout:

- **Laptop / single box** — `docker compose up` (add `-f docker-compose.gpu.yml` on an NVIDIA host). The backend entrypoint waits for Postgres, migrates, and seeds — zero manual steps.
- **Kubernetes** — `kubectl apply -k deploy/k8s/base` deploys the full stack (pgvector StatefulSet, Ollama with a model-cache PVC, backend, 2× nginx frontend, single-host ingress with `/api` routed to the backend). `deploy/k8s/overlays/gpu` schedules Ollama + the vision backend onto NVIDIA nodes and runs the LLM fully on-GPU. All manifests validate clean under `kubeconform -strict`.
- **CI/CD** — `.github/workflows/deploy.yml`: code checks → build & push both images to GHCR (tagged `latest` + commit SHA) → if a `KUBE_CONFIG_B64` secret is configured, roll the cluster deployment and wait for rollout health. A `Jenkinsfile` with the same three stages ships for teams on in-house Jenkins instead of Actions.

Scaling note: the frontend scales horizontally; the backend is pinned to 1 replica while live vision sessions and watcher threads are in-process state — the documented path to N replicas is moving session state to Redis (already in the stack for that purpose).

## Repo layout

```
simulator/          TEP process simulator (tep2py) + fault injection + permit/shift-log synthesis
models/gnn/          graph builder, GATv2 model, training, baseline threshold, attribution
rag/                 ingestion + pgvector-backed retrieval over the regulatory corpus
agents/              LangGraph state, pipeline graph, per-node logic (compound risk / permit correlation / orchestrator)
vision/              RT-DETR detectors (ppe, fall, fire_smoke, zone_intrusion), live inference, fine-tuning scripts
db/                  SQLAlchemy models, Alembic migrations, seed script, API-key auth
server/              FastAPI app tying it all together
frontend/             React + Vite dashboard (single run / time replay / live CCTV)
eval/                metrics computation against the held-out test split
scripts/             demo scenario runner, replay trace computation
docs/                scenario definitions, business-impact figures (sourced or marked as estimates)
```

## Quick start

**Linux / Windows (Docker)** — one command brings up Postgres (pgvector), Ollama, the
backend, and the frontend; migrations, seed, and the model pull happen automatically:
```bash
docker compose up            # CPU
./run-gpu.sh                 # NVIDIA GPU (needs the NVIDIA Container Toolkit)
```
Then open http://localhost:5173.

**macOS (Apple Silicon or Intel)** — Docker can't pass the Mac's Metal GPU into a
container, so on Mac we run natively via Homebrew, which *does* use the GPU for the LLM:
```bash
./setup-mac.sh               # one-time: installs deps, pulls the model, seeds the DB
./run-mac.sh                 # starts everything
```
`setup-mac.sh` needs [Homebrew](https://brew.sh). The first run installs the Python deps
(a few GB) and pulls llama3.1:8b (~5GB); after that, `./run-mac.sh` is fast.

## Running it locally (manual)

**Backend**
```bash
cd industrial-safety-intel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or the equivalent for your setup
cp .env.example .env              # set API_KEY_REQUIRED=false for local dev, or true + seed a key
alembic upgrade head
python -m db.seed                 # loads the synthetic benchmark + RAG corpus into Postgres,
                                   # prints an API key the first time it runs
uvicorn server.main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Open the printed Vite URL. If `API_KEY_REQUIRED=true`, paste the key `db.seed` printed when prompted.

Requires Postgres with the `pgvector` extension, and Ollama running locally (`ollama pull llama3.1:8b`) for incident-report generation.
