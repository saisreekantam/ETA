# Compound-risk scenario definitions

Five dual-IDV (dual-fault) scenarios, each chosen for a mechanistic "compounding" story,
not a random pairing. IDV numbers refer to the standard TEP disturbance flags. Each is
run through `tep2py` with both flags activated in the same window (controlled time offset),
plus a synthetic permit/worker-presence overlay tying it to a plant zone.

## Plant zone <-> TEP unit mapping

| Zone | TEP unit(s) | Notes |
|---|---|---|
| `reactor_zone` | Reactor | Confined-space entry point for catalyst/internals work |
| `separator_zone` | Separator | |
| `stripper_zone` | Stripper | |
| `condenser_zone` | Condenser | |
| `compressor_zone` | Recycle compressor | |
| `feed_zone` | Feed streams 1-4 | |
| `control_room` | n/a | No physical hazard; permit/worker presence not modeled here |

## Scenarios

### 1. Reactor Heat-Removal Compound Failure — `IDV4 + IDV14`
- IDV4: reactor cooling water inlet temperature (step). IDV14: reactor cooling water valve
  sticking. Individually, the control loop compensates for either one without crossing a
  naive temperature alarm threshold (confirmed empirically — see below). Combined, the
  valve can't respond fast enough to the temperature step, and reactor heat removal is
  genuinely impaired. Zone: `reactor_zone`.
- Confirmed empirically (10 Jun build): with IDV4 alone or IDV5 alone active, reactor temp
  (XMEAS9) stays within ~0.05 of baseline, but reactor cooling-water valve effort (XMV10)
  rises from ~41.9 to ~44.4 and condenser valve (XMV11) from ~19.9 to ~21.9 under the
  *compound* IDV4+IDV5 condition — a "two systems under strain at once" signature invisible
  to single-sensor thresholding on temperature alone.

### 2. Condenser/Separator Pressure Compound Failure — `IDV5 + IDV15`
- IDV5: condenser cooling water inlet temperature (step). IDV15: condenser cooling water
  valve sticking. Compounding risk: separator pressure buildup. Zone: `condenser_zone`
  (secondary: `separator_zone`).

### 3. Deferred-Maintenance + Utility-Fault Compound Risk — `IDV13 + IDV4`
- IDV13: reaction kinetics drift (slow, catalyst-aging stand-in for an overdue maintenance
  item). IDV4: reactor cooling water inlet temperature (acute utility fault). This is the
  closest TEP analogue to the problem statement's literal example — "co-occurrence of
  maintenance activity and hazardous gas accumulation" — a slow-onset degraded condition
  combined with an acute trigger. Zone: `reactor_zone`.

### 4. Feed System Compound Fault — `IDV6 + IDV1`
- IDV6: A feed loss (step). IDV1: A/C feed ratio disturbance (step). Compounding risk:
  reactor stoichiometry double-disturbed, raising runaway-reaction risk beyond what either
  alone would trigger. Zone: `feed_zone` (secondary: `reactor_zone`).

### 5. Common-Cause Utility Degradation — `IDV11 + IDV12`
- IDV11: reactor cooling water random variation. IDV12: condenser cooling water random
  variation. Modeled as simultaneous because a real common-cause failure (e.g. a cooling
  tower partial failure) would degrade both loops at once — neither loop's noise alone is
  anomalous, but synchronized degradation across both is. Zone: `reactor_zone` +
  `condenser_zone` jointly.

## Permit/worker-presence overlay rule

For each scenario, generate N=200 runs:
- ~50% "compound-positive": both IDVs activated at the scenario's onset sample, AND a
  permit (type depends on scenario — `confined_space` for #1/#3, `hot_work` for #4,
  `general` for #2/#5) scheduled to overlap the fault window in the mapped zone, AND a
  worker-presence record placing a worker in that zone during the window.
- ~50% "compound-negative" controls, split further into: (a) no faults, permit present
  (permit alone is not risk), (b) single fault only, no permit overlap, (c) single fault
  only, permit overlap (tests whether the model over-triggers on permits alone), (d)
  neither fault nor permit (pure normal operation).

## Results (multi-zone GNN, held-out test split, 30 compound-positive runs)

| Metric | GNN | Naive zone-local single-sensor baseline |
|---|---|---|
| Precision @ equal recall (1.0) | 0.909 | 0.370 |
| FNR @ 5% FPR | 0% | 80-90% |
| Top-zone localization accuracy | 100% | 23% |
| Lead time (median) | not an advantage -- see below | |

**Lead time is explicitly NOT a claimed advantage.** An earlier version of this eval let
the model see permit/worker-presence data before it was actually scheduled to appear
(a future-leak in the streaming re-inference, not in training/test labels), which made
the GNN look like it detected compound risk up to 79 minutes early. Properly time-gating
permit/presence visibility (training data was never time-gated, so the model learned
"permit visible => compound" and correctly waits for the permit to actually appear)
shows the GNN detects at roughly the same time as the baseline, sometimes later. The
defensible, headline claims are precision and zone-localization, not earlier detection
-- see `eval/results/metrics.json` for the full numbers and `eval/metrics.py`'s
`compute_lead_time` docstring for the methodology.

This overlap is the literal ground-truth compound-risk label — see `simulator/fault_injection.py`
and `simulator/permit_shiftlog_synth.py`.
