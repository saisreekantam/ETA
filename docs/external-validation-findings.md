# External Validation Findings

This document records an independent validation pass on the compound-risk GNN, run
outside the original TEP benchmark: real ICS testbed data (HAI 22.04, HAI 23.05),
classical-baseline stress tests on the original synthetic benchmark, a redesigned
synthetic benchmark that fixes a flaw the classical-baseline test exposed, and a
held-out-zone generalization test. All code lives under `eval/` (`hai_validation/`,
`tep_classical/`, `permit_fusion_benchmark/`); this doc is the narrative summary.

The short version: sensor+permit **fusion** is a real, demonstrable idea once tested
honestly. The **graph architecture specifically** is not obviously necessary for
in-distribution accuracy — simpler flat models (PCA, Random Forest) matched or beat it
repeatedly. It *is* clearly necessary for one specific thing: generalizing detection to a
zone the model never saw a labeled example of. That is the properly-supported claim.

---

## 1. Real-world validation on HAI (a real ICS testbed, not a simulation)

TEP is a simulation; HAI 22.04/23.05 are real hardware-in-the-loop testbed data
(boiler + turbine + water-treatment + auxiliary loop), freely downloadable, no permit/
human-activity layer (no public dataset has one — a structural limitation, not specific
to this project).

### 1.1 First pass (HAI 22.04) — pooled label, and why it failed

HAI only publishes one global `Attack` column, not one per subsystem. A first attempt
pooled the graph's 4 zone outputs into a single score and trained for 25 epochs with no
tuning. Result: GNN ROC-AUC ≈ **0.50** (chance level), slightly *worse* than the naive
z-score baseline (0.57). Root cause diagnosed as two compounding issues:

- Collapsing to one pooled label discards the exact per-zone structure the architecture
  is built to exploit.
- No hyperparameter tuning, 25 untuned epochs.

### 1.2 Second pass (HAI 22.04) — real per-zone labels

HAI's technical PDF documents which subsystem each of the 58 real attacks targeted.
Transcribed the catalog, cross-checked against each test file's exact attack timestamps
(counts matched exactly: 7+17+10+24=58), and rebuilt real per-zone ground truth. Retrained
with proper per-zone supervision (25 epochs):

| Metric | GNN | Baseline (z-score) |
|---|---|---|
| ROC-AUC | **0.795** | 0.616 |
| Zone localization | **93.5%** | 14.3% |
| FNR @ 20% FPR | **0.409** | 0.744 |

Clear win over the naive baseline this time — the earlier failure was a labeling/effort
problem, not evidence against the architecture.

### 1.3 Seven-way comparison — bringing in stronger baselines

Added: no-graph ablation (same GRU + zone identity, no message passing), PCA
reconstruction error, Isolation Forest, Random Forest (same features as GNN's sensor
nodes), and a hybrid (graph + PCA score fed into each zone).

| Model | ROC-AUC | Zone localization |
|---|---|---|
| Z-score baseline | 0.616 | 14.3% |
| Isolation Forest | 0.597 | 13.4% |
| No-graph ablation | 0.748 | 50.8% |
| GNN | 0.795 | 93.5% |
| Hybrid (graph + PCA) | 0.814 | 84.4% |
| PCA reconstruction (10 components, unsupervised) | 0.805 | **97.7%** |
| Random Forest (sensor features) | **0.831** | 60.3% |

Findings:
- No-graph ablation vs. GNN (93.5% vs 50.8% zone localization) shows cross-zone
  relational structure genuinely helps **localization** specifically.
- But PCA — a 10-component linear reconstruction-error baseline, fit only on normal
  data — beats the GNN on both AUC and localization. Random Forest beats it on AUC.
- Hybrid (graph + PCA feature) improved over the pure graph on AUC/F1/FNR at tight
  budgets, but *dropped* zone localization from 93.5% to 84.4% relative to the pure GNN.

**Conclusion from this pass**: the graph clearly beats a naive threshold and a
no-relational-structure ablation. It does not have a demonstrated edge over classical
methods (PCA, RF) given a fair shot, on this dataset.

### 1.4 HAI 23.05 (harder attack campaign) — the advantage didn't hold up

23.05 adds 8 new "internal-point" control-logic attacks and is documented as
"significantly more difficult to detect." Rebuilt the per-zone label catalog (52 attacks,
cross-validated against label file timestamps), retrained.

| Metric | GNN | Z-score baseline | PCA |
|---|---|---|---|
| ROC-AUC | 0.718 | 0.748 | **0.873** |
| Zone localization | 100%* | 9.3% | 15.7% |
| FNR @ 20% FPR | 0.471 | 0.453 | **0.219** |

\* Misleading: all 108 "single-zone" test windows were `P1` (the attack campaign is
documented as boiler-DCS-focused, and `P2`/`P3` had **zero** positive training windows in
this split) — the model trivially always guesses `P1` and is right by class-imbalance
accident, not cross-zone reasoning. Not a real result.

**On this second real dataset the GNN is the worst of the three on raw discrimination.**
PCA is the most consistently strong baseline across both HAI versions.

---

## 2. Classical baselines on the ORIGINAL TEP synthetic benchmark — the pivotal finding

Ran PCA, Isolation Forest, and Random Forest (sensor-only and sensor+permit+presence) on
the paper's own benchmark, using the identical train/val/test split `models/gnn/train.py`
uses, so numbers are directly comparable to the paper's reported Table 5/6.

| Model | Precision | Recall | F1 | AUC | Zone localization |
|---|---|---|---|---|---|
| GNN (paper's reported) | 1.00 | 1.00 | 1.00 | — | 100% |
| Naive z-score baseline (paper's reported) | 0.37 | 1.00 | 0.54 | — | 23.3% |
| **Random Forest, sensor-only** (no permit/presence at all) | **1.00** | **1.00** | **1.00** | **1.00** | **100%** |
| Random Forest, sensor+permit+presence | 1.00 | 1.00 | 1.00 | 1.00 | 100% |
| PCA reconstruction (sensor-only) | — | — | — | 0.762 | 20% |
| Isolation Forest (sensor-only) | — | — | — | 0.559 | 60% |

**A Random Forest given only sensor statistics — no permit, no presence, no graph —
matched the GNN's perfect score exactly.**

### Root cause

Checked `simulator/fault_injection.py`. The generator's `"compound"` condition means
**two simultaneous TEP process disturbances (IDV_a AND IDV_b)**, vs. `"single_a"`/
`"single_b"` = one disturbance. That's a strictly larger, more detectable physical
signature — not the paper's own stated premise ("a gas concentration elevated but
individually sub-alarm AND an active permit"), which implies the sensor signal alone
should be deliberately weak. A nonlinear sensor-only classifier doesn't need permit
information to separate "two things wrong" from "one thing wrong."

**Implication**: the naive z-score baseline's 0.37 precision failure was about a weak
*decision rule* (linear, single-threshold-per-channel), not a demonstration that the task
structurally requires permit fusion. The original benchmark's ground-truth construction
doesn't test what the paper's narrative says it tests.

---

## 3. A redesigned benchmark — testing the fusion premise honestly

Built a second synthetic benchmark (`eval/permit_fusion_benchmark/`) where every
condition uses the **same single fault** (same IDV, same onset distribution, same
magnitude) — the sensor trajectory is statistically identical across true-positive and
fault-having negative controls. The **only** thing that varies is whether a permit +
worker-presence record is valid **during the scored window** (last 30 of 120 samples).

Five conditions per scenario: `true_positive` (fault + permit valid now), `fp_fault_no_permit`,
`fp_fault_permit_no_overlap` (same fault, permit expired long before the scored window),
`fp_permit_no_fault`, `normal`. 1000 runs (5 scenarios × 5 conditions × 40).

Also fixed a latent bug this design required fixing: the original `build_graph` encoded
`has_permit` as "a permit record exists anywhere in the 120-sample run" (a static
whole-run boolean) rather than "a permit is currently valid" — meaning the original
architecture couldn't have used permit timing even if the labels had required it. The new
`graph_builder_v2.py` computes window-relative overlap instead, with identical feature
*shapes* (no model change needed).

A diagnostic confirmed the design worked: sensor-only RF gets AUC 1.0 telling "any fault"
from "no fault," but only 0.549 (chance) telling `true_positive` apart from the other two
fault-having conditions — sensor windows really are statistically identical, confirming no
sensor-only method can do better than chance among them.

| Model | Precision | Recall | F1 | AUC | Zone localization |
|---|---|---|---|---|---|
| z-score (naive) | 0.345 | 0.967 | 0.509 | 0.754 | 40% |
| PCA (sensor-only) | — | — | — | 0.748 | 40% |
| Isolation Forest (sensor-only) | — | — | — | 0.427 | 60% |
| Random Forest, sensor-only | 0.333 | 0.767 | 0.465 | 0.773 | 100% |
| **GNN** (sensor+permit fusion, graph) | 0.556 | 1.000 | 0.714 | 0.923 | 100% |
| **Random Forest, sensor+permit+presence (flat, no graph)** | **1.000** | **0.967** | **0.983** | **1.000** | 100% |

**The good news**: every sensor-only model is now clearly worse than every model with
permit access. The fusion premise holds when tested honestly — this is the first test in
the whole exploration where it does.

**The result that still doesn't flatter the graph**: given the *same* information (sensor
stats + permit/presence overlap) as a flat vector, plain Random Forest beats the GNN
outright (F1 0.983 vs 0.714, AUC 1.00 vs 0.92). This is a single-zone fusion problem — it
doesn't require cross-zone message passing, so GATv2's relational machinery isn't needed
to solve it.

---

## 4. Held-out-zone generalization — the decisive test

**The question this answers**: in-distribution accuracy can't show a graph is *necessary*
(a flat model is a universal approximator, given enough data). What can: whether the
model generalizes to an input configuration it was never trained on — a fixed-width flat
feature vector structurally cannot do this; a graph with parameter sharing across nodes
of the same type might.

**Design**: `s4_feed_system` (the only scenario whose positives are in `feed_zone`) was
entirely removed from train/val — no `feed_zone` example, positive or negative, during
training. At test time, the model scores `feed_zone` for the first time ever. Compared:

- **GNN** (`CompoundRiskGNN`, unmodified) — its `sensor_cluster→zone` and `permit→zone`
  GATv2 layers use the *same shared weights* regardless of which zone is on the receiving
  end.
- **Pooled Random Forest** — one shared model across all zones (not one-per-zone), with
  zone-width-agnostic pooled features + a zone one-hot + the same permit/presence context
  the GNN sees. The fairest possible non-graph competitor.

Raw threshold (0.5) precision/recall/F1 came out **0/0/0 for both models** — but this
is a calibration artifact (a never-seen zone's score scale doesn't match a threshold
tuned on other zones), not a real result. AUC (threshold-independent ranking quality) is
what matters here:

| Held-out zone | GNN AUC | Pooled RF AUC |
|---|---|---|
| `feed_zone` | **0.968** | 0.278 |
| `condenser_zone` | **0.987** | 0.264 |

Confirmed across two independent held-out zones. GNN positive-example scores cluster
around 0.15–0.33 vs. negatives at 0.01–0.05 for the held-out zone — clean separation
despite the absolute scale being uncalibrated. The pooled RF doesn't just fail to
generalize — it scores *worse than random* (some true positives get exactly 0.0 while
some negatives score above 0.2).

### Why

RF's tree splits are conditioned on the zone one-hot column, which had **zero variance**
(always 0) during training — the trees never learned a branch for it, so at test time they
fall back to leaves shaped by *other* zones' value ranges, answering a question they were
never trained on. Nothing in RF's structure enforces "the sensor+permit relationship
should mean the same thing regardless of zone" — each zone identity is just another free
input it's allowed to treat idiosyncratically.

CompoundRiskGNN's GATv2 layers are shared **per edge type**, not per zone — a hard
architectural constraint that forces one zone-agnostic transformation rule. It isn't
extrapolating in any mysterious sense; it's correctly applying a rule that was defined to
be zone-invariant from the start (the same principle that lets a CNN's shared kernel
generalize across pixel positions).

**One-line summary**: RF wins when the task is "fit the function well given examples."
The graph wins when the task is "apply the same relationship somewhere with zero
examples" — because parameter sharing across zones is architecturally guaranteed, not
learned from data.

### Caveat

Both held-out zones come from the *same* fixed 6-zone TEP topology. This shows
generalization to an unseen zone *within* a known plant structure — not to a genuinely
novel flow topology. That's the honest boundary of what's proven; extending to
cross-topology generalization would be the natural next test.

---

## 5. Overall conclusions

1. **Sensor+permit fusion is a real, demonstrable idea** — but only shown once the
   benchmark was redesigned so sensor-only information is genuinely insufficient
   (§3). The original TEP benchmark never actually tested this.
2. **Cross-zone relational reasoning has situational value** — clearly demonstrated for
   zone *localization* on HAI 22.04 (no-graph ablation: 93.5% vs 50.8%), not consistently
   for raw detection AUC across datasets.
3. **The graph is not shown to beat strong classical baselines in-distribution** — PCA
   and Random Forest matched or beat it on HAI 22.04, HAI 23.05, the original TEP
   benchmark, and even the redesigned fusion benchmark.
4. **The graph is shown to be necessary for one specific, real capability**: zero-shot
   generalization to a zone never seen in training (§4) — confirmed on two independent
   held-out zones, AUC ~0.97–0.99 vs. a flat model at ~0.26–0.28 (worse than random).

The defensible claim to carry forward is #4, not a blanket "graphs are better." The
practical implication: the value of this architecture is realized when a deployment must
generalize to zone layouts or facility configurations not exhaustively represented in
training data — not necessarily as a raw accuracy improvement on a fully-covered,
well-sampled benchmark.

## Reproducing these results

- `eval/hai_validation/` — HAI 22.04 (per-zone labels, 7-way comparison, hybrid)
- `eval/hai_validation/v2305/` — HAI 23.05
- `eval/tep_classical/` — classical baselines on the original TEP benchmark
- `eval/permit_fusion_benchmark/` — redesigned benchmark, `held_out_zone.py` for the
  generalization test (`python -m eval.permit_fusion_benchmark.held_out_zone <scenario_id>`)

Each subdirectory's scripts are runnable directly (`python -m eval.<path>.<script>`) and
write their own `results*.json` alongside the code.
