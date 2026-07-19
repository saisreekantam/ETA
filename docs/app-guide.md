# App guide (chat copilot context)

Condensed operator-facing guide to the dashboard. Loaded verbatim into the chat
assistant's context (server/chat.py) so "how do I…?" questions get real walkthroughs.
Keep it short — every token here rides along on every chat request.

## Pages (top navigation)
- **Single run** — pick a benchmark scenario in the left sidebar and run it through the
  full pipeline (GNN risk scoring → permit correlation → incident report). The plant map
  colors each zone by compound risk; hovering a zone shows which sensors drove its score;
  glowing pipes show risk propagating between zones. "Go live" streams the run
  continuously like a wallboard.
- **Time replay** — scrub a run's timeline to watch zone risk evolve sample by sample.
- **Live CCTV** — run the four hazard detectors (PPE, zone intrusion, fall, fire/smoke)
  on the sample clip, an uploaded video, a webcam, or an RTSP camera. Detections raise
  alerts in the bell inbox automatically.
- **Devices** — register cameras and IoT sensors per zone. Each sensor gets an ingest
  token at creation; push readings with:
  `curl -X POST /iot/readings -H 'X-Device-Token: <token>' -d '{"device_id":"...","value":42.5}'`
  Readings are watched automatically — a value deviating >4 sigma from the device's own
  trailing baseline raises an alert. Wrong/missing tokens raise a security alert.
- **Evaluation** — the benchmark metrics (GNN vs single-sensor baseline).
- **About** — system architecture summary.

## Other controls
- **Bell icon** — unacknowledged alerts (pipeline escalations, live CCTV detections,
  IoT anomalies, security warnings). Acknowledging records who saw it.
- **Facility chip (top right)** — click to switch facility or create a new one from an
  industry template. Benchmark scenarios exist only on the demo plant; custom
  facilities have live zones, devices, alerts, and chat.
- **Evidence button** (after a run) — printable bundle: report, citations, CCTV frame,
  audit trail.
