# Presentation Script — 7:30 target

Each section is written roughly a minute longer than its allotted slot, so you have
material to cut live rather than material you're short on. Timings in headers are the
*target* (post-trim) length, not the length of the text below it.

---

## Hook *(target 0:30)*

Industrial accidents rarely come from one thing failing. They come from a sensor
drifting, while a permit is open, while someone happens to be standing in that zone —
at the same time. Walk into any control room today and you'll find three systems
watching for exactly these three things: a SCADA system watching sensors, a
permit-to-work system watching paperwork, and CCTV watching people. Each one, on its
own, will tell you everything is fine. None of them are built to notice that all three
are true at once. That's the gap we built this to close — not another anomaly
detector, a system that reasons across sensor data, permits, and human presence
together, the way a compound incident actually happens.

---

## Motivation + industry use case *(target 1:00)*

Let's make that concrete. Picture a reactor zone in a chemical plant. A confined-space
work permit is open — legitimate, signed off. A worker is inside doing the job. And in
the background, a pressure sensor starts drifting outside its normal band — not enough
to trip a hard alarm yet, just enough to matter. Ask the SCADA system: normal, still
within soft limits. Ask the permit system: compliant, permit is valid and active. Ask
CCTV: PPE detected, no violation. Every system says green. But put those three facts
next to each other and it's not green anymore — it's exactly the setup where compound
incidents happen: DGMS and OISD incident reports are full of cases that looked
compliant in every silo and weren't compliant in combination. That's who this is for —
the control-room operator and the safety officer who currently have to hold that
correlation in their head, across three different screens, in real time, and who has
to get it right every single time to avoid it going wrong once.

---

## Approach / architecture *(target 1:00)*

So how does the system actually do that correlation. At the core is a graph neural
network — we model the plant as a heterogeneous graph: zones, sensors, permits, and
workers are all nodes, connected the way they actually relate to each other in the
plant. We use GATv2 attention so the model learns which of those relationships
actually matter for risk — a sensor node doesn't get treated the same as a permit
node, and it doesn't get treated the same in every zone either. On top of that sits a
GRU, a temporal encoder, because a single sensor reading means nothing on its own —
what matters is the trajectory, whether a value is drifting and how fast, not just
where it sits right now. That combined score per zone then feeds into an agent
pipeline built on LangGraph: a risk-scoring stage, a permit-and-PPE correlation stage
that applies severity bands to the score, and an orchestrator stage that decides what
to actually do about it — escalate, monitor, or stay quiet. Every step of that
pipeline is checkpointed, so there's a full audit trail of exactly how a risk score
turned into a decision, which matters a lot in a compliance-driven industry.

---

## Platform features *(target 1:00)*

On top of that core model, we built out the parts that make it usable by an actual
operator, not just a research result. When the orchestrator escalates something, it
doesn't just fire a number — it drafts an incident report using a local LLM, grounded
by retrieval over the real regulatory corpus, DGMS, OISD, the Factories Act, so the
report cites actual rules instead of hallucinating them. We built a knowledge-graph
chat copilot so an operator can just ask the system "why is this zone flagged" or "is
there an active permit here right now" and get an answer grounded in the live plant
state, not a canned response. We added counterfactual explanations — the system can
tell you not just that a zone is flagged, but *why*, by literally re-scoring with the
permit removed, or the worker removed, or the sensor evidence removed, and showing you
which one actually moved the needle. And we wired live CCTV detection straight into
the same orchestrator pipeline, so a vision-detected hazard gets the same reasoning
and the same escalation path as a sensor-detected one, surfaced as a real-time alert
banner the operator can't miss.

---

## Live demo *(target 2:00)*

*(Talking points to narrate over the demo — not a script to read verbatim, since this
part is driven by what's actually happening on screen.)*

- Run a compound scenario. Narrate what's happening as the map updates: "watch the
  reactor zone — it's not just flipping from green to red, it's actually moving
  through watch, then alert, then critical, tracking the model's real confidence as
  evidence accumulates."
- Point out the permit and worker-presence indicators on the map as the score climbs —
  make the audience see the three signals converging, not just a number changing.
- Trigger (or replay) a CCTV-detected hazard and let the alert banner fire — call out
  the speed ("this is seconds, not minutes") and that the reasoning text is a live LLM
  call, not a canned string.
- Click into the counterfactual explanation on the flagged zone: "watch this — if I
  remove the permit from consideration, the score barely moves. If I remove the
  worker's presence, it collapses. That's the model telling us presence is what's
  actually driving this call, not just a black box saying 'risky.'"
- If time allows, one chat-copilot question live: "why is this zone flagged" — let the
  answer come back grounded in the real data on screen.

---

## The rigor story *(target 1:30)*

Here's the part we think matters most, and the part most projects like this skip. We
didn't just trust our own benchmark. When we went back and stress-tested it with
simple classical baselines — PCA, isolation forest, random forest — we found that our
original "compound risk" label was trivially separable by a sensor-only model. No
graph needed. That's a real finding, and instead of hiding it, we treated it as the
question we actually needed to answer: is the graph architecture doing anything, or is
it dead weight. So we built a proper test for it — we held out an entire zone from
training, one the model had literally never seen a single example from, and asked it
to generalize zero-shot. The result: the GNN scores that unseen zone with an AUC of
roughly 0.97 to 0.99. A flat, non-graph model on the same held-out zone collapses to
about 0.27 — worse than random. That gap is the actual proof that the graph's shared
structure is buying something a flat model architecturally cannot get. And we didn't
stop at synthetic data either — we validated the same approach against HAI, a real
industrial control system testbed with real attack data, not just our own simulated
benchmark. This is the part of the project we're most confident defending, because
we're the ones who tried hardest to break it first.

---

## Close *(target 0:30)*

So that's the system: a graph model that reasons across sensor, permit, and human
signals the way a compound incident actually forms, wrapped in a pipeline that
explains its own decisions and cites the rules it's applying, tested against real
attack data and a generalization benchmark designed specifically to try to prove it
wrong. What's next for us is the same pipeline, live — real IoT sensor onboarding for
a new facility, not just our benchmark plant. But the core claim stands today: siloed
systems miss compound risk. This doesn't.
