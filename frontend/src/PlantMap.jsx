import { useState } from "react";
import { motion } from "motion/react";

// Continuous risk gradient (green -> amber -> orange -> red) instead of hard bands, so
// two zones at 0.55 and 0.68 read as different, not identical "medium" boxes. The
// legend's band labels still describe the same anchor points.
const RISK_STOPS = [
  [0.0, [63, 143, 114]],   // #3f8f72 normal
  [0.5, [217, 164, 65]],   // #d9a441 watch
  [0.7, [217, 130, 74]],   // #d9824a alert
  [0.9, [217, 99, 99]],    // #d96363 critical
  [1.0, [217, 99, 99]],
];

function riskColor(score) {
  if (score == null) return "url(#zone-idle-fill)";
  const s = Math.max(0, Math.min(1, score));
  for (let i = 1; i < RISK_STOPS.length; i++) {
    const [t1, c1] = RISK_STOPS[i - 1];
    const [t2, c2] = RISK_STOPS[i];
    if (s <= t2) {
      const f = (s - t1) / (t2 - t1 || 1);
      const rgb = c1.map((c, k) => Math.round(c + (c2[k] - c) * f));
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }
  }
  return "rgb(217, 99, 99)";
}

function glowFilter(score) {
  if (score == null || score < 0.5) return "none";
  const color = riskColor(score);
  return `drop-shadow(0 0 ${6 + score * 14}px ${color})`;
}

// Fixed schematic layout (overrides the DB's demo coordinates) so the map reads as an
// organised plant, not a scatter of boxes: the main process flows left-to-right across
// the middle, the recycle compressor sits on a loop above, and the control room is docked
// aside as a monitoring zone rather than on the process line. Columns are spaced evenly
// across the full canvas width so the layout fills its frame instead of bunching left.
const LAYOUT = {
  feed_zone:       { x: 26,  y: 156, w: 148, h: 92 },
  reactor_zone:    { x: 222, y: 140, w: 148, h: 118 },
  compressor_zone: { x: 222, y: 32,  w: 148, h: 80 },
  condenser_zone:  { x: 418, y: 32,  w: 148, h: 80 },
  separator_zone:  { x: 418, y: 156, w: 148, h: 92 },
  stripper_zone:   { x: 614, y: 156, w: 148, h: 92 },
  control_room:    { x: 614, y: 32,  w: 148, h: 80 },
};

// Process-flow connections drawn behind the zones, routed as right angles like a real
// plant schematic. These mirror the GNN's ZONE_FLOW_EDGES exactly (same pairs, same
// direction) -- required so the per-edge attention returned by the model maps 1:1 onto
// drawn pipes: feed -> reactor -> condenser -> separator -> stripper, with the recycle
// separator -> compressor -> reactor feeding back.
const PIPES = [
  ["feed_zone", "reactor_zone"],
  ["reactor_zone", "condenser_zone"],
  ["condenser_zone", "separator_zone"],
  ["separator_zone", "stripper_zone"],
  ["separator_zone", "compressor_zone"],
  ["compressor_zone", "reactor_zone"],
];

function center(z) {
  return { cx: z.x + z.w / 2, cy: z.y + z.h / 2 };
}

// Right-angle connector that starts and ends on the BOX EDGES facing each other, so the
// pipe tucks into the equipment rather than crossing over a box face. Pipes are also drawn
// before the boxes (see render order) so any overlap sits underneath.
function orthPath(a, b) {
  const ca = center(a), cb = center(b);
  const sameRow = Math.abs(ca.cy - cb.cy) < 30;
  const sameCol = Math.abs(ca.cx - cb.cx) < 30;

  if (sameRow) {
    // exit the right/left edge of each box at the shared centre height
    const leftBox = ca.cx < cb.cx ? a : b;
    const rightBox = ca.cx < cb.cx ? b : a;
    const y = (center(leftBox).cy + center(rightBox).cy) / 2;
    return `M ${leftBox.x + leftBox.w} ${y} L ${rightBox.x} ${y}`;
  }
  if (sameCol) {
    // exit the top/bottom edge of each box at the shared centre x
    const topBox = ca.cy < cb.cy ? a : b;
    const botBox = ca.cy < cb.cy ? b : a;
    const x = (center(topBox).cx + center(botBox).cx) / 2;
    return `M ${x} ${topBox.y + topBox.h} L ${x} ${botBox.y}`;
  }
  // offset: leave the side of A facing B at A's centre height, run to a mid-x that sits
  // in the actual gap between the two boxes' facing edges, then drop/rise into the
  // top/bottom centre of B. Because midX is between A's near edge and B's near edge (not
  // inside either box's own x-range), the final horizontal-into-B segment starts outside
  // B's footprint and enters perpendicular to its edge, instead of skimming across it.
  const aRight = cb.cx > ca.cx;
  const startX = aRight ? a.x + a.w : a.x;
  const startY = ca.cy;
  const endX = cb.cx;
  const endY = cb.cy < ca.cy ? b.y + b.h : b.y;
  const bNearEdge = aRight ? b.x : b.x + b.w;
  const midX = (startX + bNearEdge) / 2;
  return `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`;
}

const PERMIT_SHORT = { hot_work: "HOT WORK", confined_space: "CONFINED", electrical: "ELECTRICAL", general: "GENERAL" };

// Hover tooltip: mini bar chart of the gradient-saliency behind a zone's score --
// "which sensors drove this number" -- rendered inside the SVG so it tracks the zone
// boxes without any portal/positioning machinery.
function SaliencyTooltip({ zone, saliency, width, height }) {
  const rows = saliency.slice(0, 5);
  const boxW = 200;
  const boxH = 34 + rows.length * 17;
  const x = zone.x + zone.w + 10 + boxW > width ? zone.x - boxW - 10 : zone.x + zone.w + 10;
  const y = Math.max(6, Math.min(zone.y, height - boxH - 6));
  const barMax = 108;

  return (
    <g className="saliency-tip" pointerEvents="none">
      <rect x={x} y={y} width={boxW} height={boxH} rx={8} className="saliency-tip-bg" />
      <text x={x + 10} y={y + 17} className="saliency-tip-title">Top sensors · gradient saliency</text>
      {rows.map((r, i) => {
        const rowY = y + 30 + i * 17;
        return (
          <g key={r.sensor}>
            <text x={x + 10} y={rowY + 8} className="saliency-tip-sensor">{r.sensor}</text>
            <rect x={x + 78} y={rowY} width={barMax} height={10} rx={3} className="saliency-tip-track" />
            <rect x={x + 78} y={rowY} width={Math.max(3, r.saliency * barMax)} height={10} rx={3}
                  className="saliency-tip-bar" />
            <text x={x + 78 + barMax + 4} y={rowY + 8.5} className="saliency-tip-pct" textAnchor="start">
              {Math.round(r.saliency * 100)}%
            </text>
          </g>
        );
      })}
    </g>
  );
}

export default function PlantMap({ zones, riskByZone, baselineByZone, activeZone, permits, workerPresence, sensorSaliencyByZone, flowAttention }) {
  const [hoveredZone, setHoveredZone] = useState(null);
  if (!zones) {
    return (
      <div style={{ height: 280, display: "flex", alignItems: "center", justifyContent: "center", color: "#565f73", fontSize: 13 }}>
        Loading plant layout…
      </div>
    );
  }
  const width = 788;
  const height = 280;

  // The hardcoded schematic LAYOUT (and its process-flow pipes) is the demo plant's.
  // Custom facilities render from their DB zone coordinates instead -- template zones
  // (server/facilities.py) are laid out on this same canvas. All-or-nothing so a
  // custom facility that reuses a demo key (e.g. control_room) doesn't get a mix of
  // the two coordinate systems.
  const isDemoLayout = Object.keys(zones).length > 0 && Object.keys(zones).every((k) => LAYOUT[k]);
  const boxFor = (zoneId) => (isDemoLayout ? LAYOUT[zoneId] : zones[zoneId]);

  // Active permits and on-shift workers per zone -- the "permit overlaps + worker
  // location" situational-awareness layer over the risk heat.
  const activePermitByZone = {};
  for (const p of permits || []) {
    if (p.status === "active" && !activePermitByZone[p.zone]) activePermitByZone[p.zone] = p;
  }
  const workersByZone = {};
  for (const w of workerPresence || []) {
    // Benchmark records carry has_presence over a run window; live records would have a
    // null exit_time while the worker is still inside. Either counts as "in the zone".
    const present = w.has_presence != null ? w.has_presence : !w.exit_time;
    if (present) workersByZone[w.zone] = (workersByZone[w.zone] || 0) + 1;
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="plant-map" preserveAspectRatio="xMidYMid meet">
      <defs>
        <pattern id="permit-hatch" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
          <rect width="7" height="7" fill="none" />
          <line x1="0" y1="0" x2="0" y2="7" stroke="rgba(255,255,255,0.35)" strokeWidth="1.6" />
        </pattern>
        {/* glass sheen laid over every zone box: bright top edge fading out, like light
            catching a pane over the process floor */}
        <linearGradient id="zone-glass" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffffff" stopOpacity="0.22" />
          <stop offset="0.35" stopColor="#ffffff" stopOpacity="0.05" />
          <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
        </linearGradient>
        {/* Idle equipment (no score yet): a subtle top-lit fill instead of flat color,
            so unscored zones read as real hardware on the floor, not empty placeholders. */}
        <linearGradient id="zone-idle-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="var(--zone-null-fill-2)" />
          <stop offset="1" stopColor="var(--zone-null-fill)" />
        </linearGradient>
      </defs>

      <g className="pipes">
        {isDemoLayout && PIPES.map(([a, b]) => {
          const za = LAYOUT[a], zb = LAYOUT[b];
          if (!za || !zb) return null;
          const d = orthPath(za, zb);
          // GATv2 attention on this process-flow edge, scaled by the source zone's
          // risk: a pipe glows hard when a risky upstream zone is what its neighbor is
          // attending to -- risk visibly propagating through the plant, which is the
          // whole argument for the graph model. (Attention normalizes over each
          // target's incoming edges, so source risk supplies the magnitude.)
          const att = flowAttention?.find((f) => (f.src === a && f.dst === b) || (f.src === b && f.dst === a));
          const srcRisk = att ? Math.max(riskByZone[att.src] || 0, 0) : 0;
          const w = att ? att.attention * (0.2 + 0.8 * srcRisk) : null;
          // Risk propagating along this edge is drawn as a flowing overlay whose width,
          // colour and glow scale with attention-weighted risk. Gated on w > 0.15 so the
          // animation only appears where there is actual signal, never as idle decoration.
          return (
            <g key={`${a}-${b}`}>
              <path className="zone-pipe" d={d} fill="none"
                    style={w != null ? { strokeWidth: 2 + w * 5 } : undefined} />
              {w != null && w > 0.15 && (
                <path className="zone-pipe-flow" d={d} fill="none"
                      style={{
                        strokeWidth: 1.5 + w * 5.5,
                        opacity: 0.35 + Math.min(w, 1) * 0.65,
                        stroke: w > 0.45 ? "var(--accent-red)" : w > 0.2 ? "var(--accent-amber)" : "var(--accent-cyan)",
                        filter: `drop-shadow(0 0 ${3 + w * 9}px currentColor)`,
                      }} />
              )}
            </g>
          );
        })}
      </g>

      {Object.keys(zones).filter((zoneId) => boxFor(zoneId)).map((zoneId) => {
        const z = boxFor(zoneId);
        const label = zones[zoneId].label;
        const score = riskByZone[zoneId];
        const baseline = baselineByZone[zoneId];
        const isActive = zoneId === activeZone;
        const isCritical = isActive && score >= 0.9;
        const permit = activePermitByZone[zoneId];
        const workers = workersByZone[zoneId] || 0;

        const saliency = (sensorSaliencyByZone && sensorSaliencyByZone[zoneId]) || [];

        return (
          <motion.g
            key={zoneId}
            initial={false}
            style={{ filter: glowFilter(isActive ? score : null) }}
            onMouseEnter={saliency.length ? () => setHoveredZone(zoneId) : undefined}
            onMouseLeave={saliency.length ? () => setHoveredZone(null) : undefined}
          >
            <motion.rect
              x={z.x} y={z.y} width={z.w} height={z.h}
              rx={12}
              initial={false}
              animate={{
                fill: riskColor(score),
                stroke: isActive ? "#ffffff" : "var(--border-strong)",
                strokeWidth: isActive ? 2.5 : 1.5,
                opacity: score == null ? 0.5 : 0.92,
                scale: isActive ? 1.015 : 1,
              }}
              style={{ transformOrigin: `${z.x + z.w / 2}px ${z.y + z.h / 2}px` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            />
            <rect x={z.x} y={z.y} width={z.w} height={z.h} rx={12}
                  fill="url(#zone-glass)" pointerEvents="none" />
            {permit && score != null && (
              <rect x={z.x} y={z.y} width={z.w} height={z.h} rx={12}
                    fill="url(#permit-hatch)" pointerEvents="none" />
            )}
            {isCritical && (
              <motion.rect
                x={z.x} y={z.y} width={z.w} height={z.h}
                rx={12}
                fill="none"
                stroke="#fb5858"
                strokeWidth={2}
                animate={{ opacity: [0.9, 0, 0.9], scale: [1, 1.08, 1] }}
                style={{ transformOrigin: `${z.x + z.w / 2}px ${z.y + z.h / 2}px` }}
                transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
              />
            )}
            {permit && score != null && (
              <g>
                <rect x={z.x + 6} y={z.y + 6} rx={4} width={PERMIT_SHORT[permit.permit_type].length * 5.4 + 12} height={14}
                      fill="rgba(8,11,17,0.55)" />
                <text x={z.x + 12} y={z.y + 16.5} className="zone-permit-tag">
                  {PERMIT_SHORT[permit.permit_type]}
                </text>
              </g>
            )}
            <text x={z.x + z.w / 2} y={z.y + z.h / 2 - (score != null ? 8 : 0)} textAnchor="middle" className="zone-label">
              {label}
            </text>
            {score != null && (
              <>
                <text x={z.x + z.w / 2} y={z.y + z.h / 2 + 12} textAnchor="middle" className="zone-score">
                  {score.toFixed(2)}
                </text>
                {baseline != null && (
                  <text x={z.x + z.w / 2} y={z.y + z.h / 2 + 25} textAnchor="middle" className="zone-score-sub">
                    base {baseline.toFixed(2)}
                  </text>
                )}
              </>
            )}
            {workers > 0 && score != null && (
              <g>
                {Array.from({ length: Math.min(workers, 5) }).map((_, i) => (
                  <circle key={i} cx={z.x + z.w - 12 - i * 11} cy={z.y + z.h - 11} r={4}
                          className="zone-worker-dot" />
                ))}
                {workers > 5 && (
                  <text x={z.x + z.w - 12 - 5 * 11 - 4} y={z.y + z.h - 8} textAnchor="end" className="zone-worker-count">
                    +{workers - 5}
                  </text>
                )}
                <title>{`${workers} worker(s) present`}</title>
              </g>
            )}
          </motion.g>
        );
      })}

      {hoveredZone && boxFor(hoveredZone) && (sensorSaliencyByZone?.[hoveredZone]?.length > 0) && (
        <SaliencyTooltip
          zone={boxFor(hoveredZone)}
          saliency={sensorSaliencyByZone[hoveredZone]}
          width={width}
          height={height}
        />
      )}
    </svg>
  );
}
