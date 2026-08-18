import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { BarChart3, Crosshair, Gauge, MapPin, Network, Timer } from "lucide-react";
import { getBenchmarks, getGeneralizationBenchmarks } from "./api";

/** GNN = chromatic (accent blue), baseline = achromatic gray: the chroma difference
 * survives every CVD type, and every bar is direct-labeled, so identity never rides on
 * color alone. */
const SERIES = [
  { key: "gnn", label: "Compound-risk GNN", color: "var(--accent-cyan)" },
  { key: "baseline", label: "Single-sensor baseline", color: "var(--text-tertiary)" },
];

function Legend() {
  return (
    <div className="bench-legend">
      {SERIES.map((s) => (
        <span key={s.key} className="legend-item">
          <span className="legend-swatch" style={{ background: s.color }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** Horizontal paired bars for one metric group. values: [{label, gnn, baseline}],
 * formatted with fmt; bars are thin with rounded data-ends and a value label at the end. */
function PairedBars({ rows, fmt = (v) => v.toFixed(2), max = 1 }) {
  const barH = 14, gap = 6, groupGap = 18, labelW = 118, valueW = 52;
  const chartW = 460;
  const plotW = chartW - labelW - valueW;
  const groupH = barH * 2 + gap;
  const height = rows.length * (groupH + groupGap) - groupGap + 4;

  return (
    <svg className="bench-chart" viewBox={`0 0 ${chartW} ${height}`} role="img">
      {rows.map((row, gi) => {
        const y0 = gi * (groupH + groupGap);
        return (
          <g key={row.label}>
            <text x={labelW - 10} y={y0 + groupH / 2 + 4} textAnchor="end" className="bench-row-label">
              {row.label}
            </text>
            {SERIES.map((s, si) => {
              const v = row[s.key];
              const w = Math.max(2, (v / max) * plotW);
              const y = y0 + si * (barH + gap);
              return (
                <g key={s.key}>
                  <title>{`${s.label} · ${row.label}: ${fmt(v)}`}</title>
                  <rect x={labelW} y={y} width={w} height={barH} rx={4}
                        fill={s.color} opacity={0.92} />
                  <text x={labelW + w + 8} y={y + barH - 3} className="bench-value">{fmt(v)}</text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

const GEN_SERIES = [
  { key: "gnn", label: "Compound-risk GNN", color: "var(--accent-cyan)" },
  { key: "rf", label: "Pooled Random Forest (same features, no graph)", color: "var(--text-tertiary)" },
];

/** Same paired-bar rendering as PairedBars, but its own series/labels -- the competitor
 * here is a pooled flat classifier given identical features, not the naive single-sensor
 * threshold the rest of this page compares against, so it needs its own legend. */
function GenBars({ rows }) {
  const barH = 14, gap = 6, groupGap = 18, labelW = 118, valueW = 52;
  const chartW = 460;
  const plotW = chartW - labelW - valueW;
  const groupH = barH * 2 + gap;
  const height = rows.length * (groupH + groupGap) - groupGap + 4;

  return (
    <svg className="bench-chart" viewBox={`0 0 ${chartW} ${height}`} role="img">
      {rows.map((row, gi) => {
        const y0 = gi * (groupH + groupGap);
        return (
          <g key={row.label}>
            <text x={labelW - 10} y={y0 + groupH / 2 + 4} textAnchor="end" className="bench-row-label">
              {row.label}
            </text>
            {GEN_SERIES.map((s, si) => {
              const v = row[s.key];
              const w = Math.max(2, v * plotW);
              const y = y0 + si * (barH + gap);
              return (
                <g key={s.key}>
                  <title>{`${s.label} · ${row.label}: AUC ${v.toFixed(3)}`}</title>
                  <rect x={labelW} y={y} width={w} height={barH} rx={4} fill={s.color} opacity={0.92} />
                  <text x={labelW + w + 8} y={y + barH - 3} className="bench-value">{v.toFixed(2)}</text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function GeneralizationSection() {
  const [data, setData] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getGeneralizationBenchmarks().then(setData).catch(() => setFailed(true));
  }, []);

  if (failed || !data) return null; // optional section -- absence shouldn't break the page

  const rows = Object.values(data.runs).map((r) => ({
    label: r.held_out_zone,
    gnn: r.gnn.auc,
    rf: r.pooled_random_forest.auc,
  }));
  if (rows.length === 0) return null;

  return (
    <section className="bench-card">
      <h3><Network size={16} style={{ verticalAlign: "-2px", marginRight: 6 }} />
        Why the graph, not just a threshold
      </h3>
      <p className="devices-lede">
        The table above shows the GNN beating a single-sensor threshold — but classical
        methods (PCA, Random Forest) given the same features match or beat the GNN on
        that same in-distribution benchmark. The graph's real advantage shows up here:
        each zone below was held out of training entirely — zero examples, positive or
        negative — and scored for the first time at test. AUC (ranking quality) is the
        honest metric; a 0.5-threshold precision/recall is meaningless for a zone the
        model has never been calibrated on.
      </p>
      <div className="bench-charts" style={{ gridTemplateColumns: "1fr" }}>
        <GenBars rows={rows} />
      </div>
      <p className="bench-note">
        The GNN's per-edge-type weights are shared across every zone, so the sensor+permit
        fusion rule it learns on other zones transfers automatically. The pooled Random
        Forest, given identical information, has no such constraint — its tree splits are
        tied to the specific zones it trained on, and it scores worse than random on a zone
        it's never seen (AUC below 0.5 on both held-out zones tested).
      </p>
    </section>
  );
}

function StatTile({ icon: Icon, title, gnn, baseline, note }) {
  return (
    <div className="bench-tile">
      <div className="bench-tile-head"><Icon size={14} /> {title}</div>
      <div className="bench-tile-values">
        <div className="bench-tile-main">
          <span className="bench-big">{gnn}</span>
          <span className="bench-who">GNN</span>
        </div>
        <div className="bench-tile-sub">
          <span className="bench-small">{baseline}</span>
          <span className="bench-who">baseline</span>
        </div>
      </div>
      {note && <div className="bench-tile-note">{note}</div>}
    </div>
  );
}

export default function Evaluation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getBenchmarks().then(setData).catch((e) => setError(String(e.message || e)));
  }, []);

  if (error) return <div className="status-banner error">{error}</div>;
  if (!data) return <div className="status-banner loading"><span className="spinner" /> Loading evaluation results…</div>;

  const acc = data.accuracy_vs_baseline;
  const fnr = data.fnr_at_matched_fpr;
  const zone = data.zone_localization;
  const lead = data.lead_time;

  const prfRows = [
    { label: "Precision", gnn: acc.gnn.precision, baseline: acc.baseline_single_sensor.precision },
    { label: "Recall", gnn: acc.gnn.recall, baseline: acc.baseline_single_sensor.recall },
    { label: "F1", gnn: acc.gnn.f1, baseline: acc.baseline_single_sensor.f1 },
  ];

  const fnrRows = fnr.gnn.map((g, i) => ({
    label: `at ${Math.round(g.target_fpr * 100)}% FPR`,
    gnn: g.fnr,
    baseline: fnr.baseline_single_sensor[i]?.fnr ?? 0,
  }));

  return (
    <div className="bench-page">
      <div className="bench-header">
        <h2><BarChart3 size={19} /> Evaluation vs single-sensor baseline</h2>
        <p className="devices-lede">
          Held-out test split: {acc.n_test_runs} runs, {acc.n_compound_positive} compound-positive,
          scored at each run's true hazard zone. Computed offline by <code>eval/metrics.py</code> —
          these are the judged numbers, presented from the same artifact.
        </p>
      </div>

      <div className="bench-tiles">
        <StatTile icon={Crosshair} title="False-negative rate @ 5% FPR"
                  gnn={`${Math.round(fnr.gnn[0].fnr * 100)}%`}
                  baseline={`${Math.round(fnr.baseline_single_sensor[0].fnr * 100)}%`}
                  note="Missed hazards at a matched false-alarm budget — the metric that saves lives." />
        <StatTile icon={MapPin} title="Zone localization"
                  gnn={`${Math.round(zone.gnn_top_zone_accuracy * 100)}%`}
                  baseline={`${Math.round(zone.baseline_top_zone_accuracy * 100)}%`}
                  note={`Top-scoring zone matches the true fault zone (${zone.n_compound_test_runs} compound runs).`} />
        <StatTile icon={Gauge} title="F1 at true zone"
                  gnn={acc.gnn.f1.toFixed(2)}
                  baseline={acc.baseline_single_sensor.f1.toFixed(2)}
                  note={`Precision ${acc.gnn.precision.toFixed(2)} vs ${acc.baseline_single_sensor.precision.toFixed(2)} — the baseline alarms constantly to hit the same recall.`} />
        <StatTile icon={Timer} title="Median confirm latency"
                  gnn={`${lead.gnn.median_minutes.toFixed(0)} min`}
                  baseline={`${lead.baseline_single_sensor.median_minutes.toFixed(0)} min`}
                  note="After ground-truth onset. The baseline fires at onset but at 37% precision; the GNN confirms with high precision and the right zone." />
      </div>

      <div className="bench-charts">
        <section className="bench-card">
          <h3>Detection quality at the true hazard zone</h3>
          <Legend />
          <PairedBars rows={prfRows} />
        </section>

        <section className="bench-card">
          <h3>False-negative rate at matched false-positive budgets</h3>
          <Legend />
          <PairedBars rows={fnrRows} fmt={(v) => `${Math.round(v * 100)}%`} />
          <p className="bench-note">
            Lower is better. At every false-alarm budget the baseline misses 60–90% of real
            compound hazards; the GNN misses none on this split.
          </p>
        </section>
      </div>

      <section className="bench-card">
        <h3>All numbers</h3>
        <table className="bench-table">
          <thead>
            <tr><th>Metric</th><th>Compound-risk GNN</th><th>Single-sensor baseline</th></tr>
          </thead>
          <tbody>
            <tr><td>Precision</td><td>{acc.gnn.precision}</td><td>{acc.baseline_single_sensor.precision}</td></tr>
            <tr><td>Recall</td><td>{acc.gnn.recall}</td><td>{acc.baseline_single_sensor.recall}</td></tr>
            <tr><td>F1</td><td>{acc.gnn.f1}</td><td>{acc.baseline_single_sensor.f1}</td></tr>
            {fnrRows.map((r) => (
              <tr key={r.label}><td>FNR {r.label}</td><td>{Math.round(r.gnn * 100)}%</td><td>{Math.round(r.baseline * 100)}%</td></tr>
            ))}
            <tr><td>Zone localization</td><td>{Math.round(zone.gnn_top_zone_accuracy * 100)}%</td><td>{Math.round(zone.baseline_top_zone_accuracy * 100)}%</td></tr>
            <tr><td>Median confirm latency</td><td>{lead.gnn.median_minutes} min</td><td>{lead.baseline_single_sensor.median_minutes} min</td></tr>
          </tbody>
        </table>
        <p className="bench-note">{lead.note}</p>
      </section>

      <GeneralizationSection />
    </div>
  );
}
