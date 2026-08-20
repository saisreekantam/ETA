# Pitch deck generation prompt

Paste everything below into Genspark / Claude (or any AI slide generator) as one prompt.
It's a complete slide-by-slide brief: content, speaker notes, and what to draw on each
slide, so the generator doesn't have to invent facts.

---

## PROMPT STARTS HERE

Generate a 10-slide pitch deck for a hackathon presentation (7–8 minute pitch + a
separate live demo, so slides should be light on text — support the talk, don't read
off the slide). Follow the design brief and per-slide spec exactly; do not invent
numbers, features, or claims beyond what's given below.

### Design brief (apply to every slide)

- **Palette — light, not corporate-blue-on-white**: background a warm off-white
  (`#FAF8F3` or similar), primary text a dark warm gray (`#2A2620`), NOT pure black.
  One accent family only, borrowed from the product's own risk gradient so the deck
  visually matches the live demo: soft green `#3F8F72` (safe/normal), amber `#D9A441`
  (watch), orange `#D9824A` (alert), muted red `#D96363` (critical). Use the green as
  the default accent for headers/highlights; reserve amber→red only for risk-related
  visuals (the architecture and validation slides), never as decorative color.
- **Typography**: one clean sans-serif (e.g. Inter, Söhne, or system sans), large
  headline (36–44pt), sparse body text (max 4 bullets per slide, 6–8 words each where
  possible). No paragraphs on slides — the paragraph is the speaker note, not the slide.
- **Layout**: generous whitespace, left-aligned content, diagrams centered or
  right-aligned with text on the left. No slide should feel dense — if a slide needs
  more than 4 bullets + 1 diagram, split it.
- **Diagrams**: simple flat boxes-and-arrows style, rounded corners, thin strokes, using
  only the accent palette above — not skeuomorphic, not 3D, not clip-art icons.

---

### Slide 1 — Title

**Content:**
- Project name / team name (placeholder — fill in)
- One-line tagline: **"Catching the risk that no single system sees alone."**
- Hackathon name + date (placeholder)

**Visual:** minimal — just the tagline, no diagram. Optional: a faint background
schematic of connected plant zones (feed → reactor → condenser → separator →
stripper/compressor) at very low opacity, in the green accent.

**Speaker note (not on slide):** Don't read the title — say the tagline as your
opening line and move straight to slide 2.

---

### Slide 2 — The problem (Hook)

**Headline:** "Every system says normal. Together, they're not."

**Content (3 bullets, one per silo):**
- Sensors watch equipment
- Permits watch paperwork
- Cameras watch people
- *(4th line, visually separated/bolded)* None of them watch all three together.

**Visual:** three simple icons/boxes (Sensor / Permit / Camera) each glowing green
("normal") with a large red question mark or exclamation where the three would need to
overlap — the visual should make "each one is fine alone" obvious at a glance.

**Speaker note:** Industrial accidents rarely come from one failure. They come from a
sensor drifting, while a permit is open, while someone happens to be standing in that
zone — at the same time. Every existing system checks its own silo and reports normal.
None of them are built to notice all three are true at once.

---

### Slide 3 — Motivation + industry use case (merged)

**Headline:** "One scenario, three silos, three green lights"

**Content (walk one concrete case as 4 short lines):**
- Reactor zone · confined-space permit open, signed off
- Worker inside, doing the job
- Pressure sensor drifting — not yet alarm-level
- SCADA: normal · Permits: compliant · CCTV: PPE detected — **all green**

**Visual:** a single "zone card" mockup (styled like the actual plant-map zone boxes
from the live product) showing the reactor zone with a permit tag, a worker icon, and a
sensor reading — colored amber/watch rather than red, to visually foreshadow the
gradient the audience will see in the live demo.

**Speaker note:** This is exactly the setup where compound incidents happen — DGMS and
OISD incident reports are full of cases that looked compliant in every silo and weren't
compliant in combination. This is who it's for: the control-room operator and safety
officer who today have to hold that correlation in their head, across three screens, in
real time.

---

### Slide 4 — Approach / architecture

**Headline:** "A graph that reasons across sensor, permit, and person"

**Content (label only, let the diagram carry it):**
- Heterogeneous graph: zones, sensors, permits, workers
- GATv2 attention — what matters, and where
- GRU — how it's trending over time
- LangGraph pipeline: risk → correlation → orchestration

**Visual — the architecture diagram (be precise, this is the most technical slide):**
```
[Raw sensor window, 30 steps] --> [GRU] --> [zone-cluster embedding]
                                                    |
[sensor / permit / worker / zone nodes] -----------+
                                                    v
                                        [GATv2 layer 1] --> [GATv2 layer 2]
                                                    |
                                                    v
                                    [per-zone risk score, 7 zones]
                                                    |
                                                    v
                        [Risk scoring] -> [Permit/PPE correlation] -> [Orchestrator: RAG + LLM]
```
Render as a clean left-to-right flow diagram, boxes in the light palette, one arrow
color (dark warm gray), GRU and GATv2 boxes highlighted in the green accent since
they're the two core model components.

**Speaker note:** We model the plant as a heterogeneous graph — zones, sensors,
permits, workers as nodes, connected the way they actually relate in the plant. A GRU
reads each zone's own raw sensor trend first — not just the current value, the
trajectory. GATv2 attention then lets that trend information propagate across the
plant, so risk in one zone can influence its neighbors. That feeds a LangGraph
pipeline: risk scoring, then permit/PPE correlation, then an orchestrator that decides
what to actually do about it.

---

### Slide 5 — Platform features

**Headline:** "Built to be used, not just benchmarked"

**Content (4 bullets, one feature each — icon + short label, not sentences):**
- **Grounded reports** — LLM cites real regulation text (DGMS / OISD / Factories Act), never hallucinated
- **Knowledge-graph copilot** — ask "why is this zone flagged," get a live, grounded answer
- **Counterfactual explanations** — "would this still flag without the permit / the worker?"
- **CCTV → orchestrator** — a camera-detected hazard gets the same reasoning path as a sensor one

**Visual:** four small equal-weight cards in a 2×2 grid, one icon per feature (document/
citation icon, chat-bubble icon, branching-arrow icon, camera icon), all in the green
accent, no hierarchy between them — this slide is about breadth, not depth.

**Speaker note:** Keep this fast — one sentence per feature, don't over-explain any
single one, save the depth for the live demo.

---

### Slide 6 — Built for production, not just a benchmark

**Headline:** "The parts that make it usable on a real shift"

**Content — two columns, 3 lines each:**

*Operator experience:*
- **Break mode** ("tea-time helper") — step away safely; an emergency still reaches your phone
- **Live alert banner** — a real-time on-screen alert, not just a bell icon nobody checks
- **One-click CSV export** — every alert, timestamp, and sensor value, for offline analysis

*Infrastructure:*
- **Pluggable IoT ingestion** — any device that can POST JSON, token-authenticated per device
- **Dockerized end-to-end**, deployed via Kubernetes (GKE + generic kubeconfig)
- **CI/CD via GitHub Actions** — build, test, and roll out on every push

**Visual:** two-column layout, a small icon per line (coffee-cup icon for break mode,
bell/banner icon, download icon / Docker whale icon, Kubernetes wheel icon, CI arrow-
loop icon) — all in the green accent, consistent weight, no single item emphasized over
another.

**Speaker note:** This slide exists to answer the unspoken question — "is this a script
that only runs on your laptop, or something you could actually hand to a plant?" It's
containerized, it deploys to Kubernetes, it has CI/CD, and it has the operator-facing
details — a break button, a real alert banner, exportable data — that a research
prototype usually skips.

---

### Slide 7 — Live demo (transition slide)

**Headline:** "Let's watch it happen" *(or similar — minimal text)*

**Content:** none, or a single line: "Live demo — compound scenario, CCTV alert,
counterfactual explanation"

**Visual:** full-bleed, either blank/dark-adjacent (to visually cue "we're leaving the
slides now") or a large static screenshot of the plant map mid-alert (amber/orange
zone) as a background, dimmed, with the headline text on top.

**Speaker note:** This is your cue slide — say almost nothing here, switch to the live
app. (Demo beats: run a compound scenario and narrate the map moving through
watch→alert→critical; trigger a CCTV alert and call out the speed; click a
counterfactual explanation and point out which factor actually drove the score.)

---

### Slide 8 — The rigor story (validation)

**Headline:** "We tried hardest to prove ourselves wrong"

**Content (short, number-forward):**
- Found our own benchmark's "compound" label was trivially separable by a sensor-only model
- Built a held-out-zone test: score a zone the model never trained on
- **GNN: ~0.97–0.99 AUC** on the unseen zone
- **Flat model: ~0.27 AUC** — worse than random
- Also validated against HAI, a real industrial-control-system attack testbed

**Visual — a simple horizontal bar chart** (this is the one slide that should carry a
real chart, not just a diagram):
- Two bars: "Graph model" (green, ~0.97–0.99) vs "Flat / non-graph model" (muted red,
  ~0.27), x-axis labeled "AUC on a zone never seen in training, 0 to 1"
- Keep it to just these two bars — no legend clutter, label each bar directly with its
  value.

**Speaker note:** We didn't just trust our own benchmark — we stress-tested it with
classical baselines and found a real weakness, then built a proper test to answer the
question honestly: is the graph structure doing anything? The held-out-zone result is
the answer — that gap is proof the shared graph structure buys something a flat model
architecturally cannot get.

---

### Slide 9 — Real-world results (HAI industrial testbed)

**Headline:** "Real attack data — and an honest scorecard"

**Content (be precise with numbers, this slide's credibility depends on it):**
- Validated on HAI, a real industrial-control-system testbed with real attacks — not
  our own synthetic benchmark — across two independent releases (HAI 22.04, 23.05)
- **Where classical/rule-based methods hold their own:** raw anomaly-detection AUC —
  PCA and Random Forest matched or beat our model (≈0.80–0.83 vs our ≈0.72–0.79).
  Real ICS data is genuinely hard, and simple unsupervised baselines are a legitimately
  strong first line of defense.
- **Where the graph model wins, consistently, on both releases:** zone
  localization — correctly identifying *which* zone is under attack, not just that
  something is wrong.
- **Structural point:** none of the classical baselines can take a permit or a
  worker's presence as input at all — they only ever see sensor values. Ours is the
  only architecture here built to reason about all three together.

**Visual — grouped bar chart, "zone localization accuracy":**
- X-axis: two groups, "HAI 22.04" and "HAI 23.05"
- Each group: two bars — "Single-sensor rule" (muted red) vs "Our graph model" (green)
- Values: HAI 22.04 — 14.3% vs 93.5%. HAI 23.05 — 9.3% vs 100%.
- Label each bar directly with its percentage; keep axis simple (0–100%).

**Speaker note:** We're not going to stand up here and say we beat every classical
method on every metric on real attack data — we didn't, and pretending otherwise would
undercut the rigor story. On raw anomaly AUC, PCA and Random Forest are genuinely
competitive, sometimes ahead. But across two independent real-world releases, our
model's zone-localization is dramatically and consistently better than a single-sensor
rule — and it's the only one of these models that can even see a permit or a worker in
the first place, which is the entire point of what we're building.

---

### Slide 10 — Close

**Headline:** "Siloed systems miss compound risk. This doesn't."

**Content:**
- One line on what's next: real sensor onboarding for a live facility
- Thank-you / contact / repo link (placeholder)

**Visual:** return to the same faint plant-schematic motif from slide 1, now lit up
green end-to-end, for a visual bookend.

**Speaker note:** Land back on the opening thesis, then stop talking — leave room for
questions.

---

### Optional backup slides (only if asked for appendix material)

- **Backup A — Dataset**: how the synthetic benchmark is generated (TEP simulator +
  fault injection + permit/presence overlay with anti-shortcut negative controls).
- **Backup B — Escalation logic**: the exact severity/escalation bands
  (0.5/0.7/0.9 → medium/high/critical, monitor/alert/emergency) as a simple table, to
  show the rule layer is auditable, not a black box.

## PROMPT ENDS HERE
