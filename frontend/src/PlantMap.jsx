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
  if (score == null) return "var(--zone-null-fill)";
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
// aside as a monitoring zone rather than on the process line. Coordinates are on the
// 820x360 canvas below.
const LAYOUT = {
  feed_zone:       { x: 24,  y: 150, w: 140, h: 92 },
  reactor_zone:    { x: 196, y: 134, w: 140, h: 118 },
  separator_zone:  { x: 368, y: 150, w: 140, h: 92 },
  stripper_zone:   { x: 540, y: 150, w: 130, h: 92 },
  condenser_zone:  { x: 368, y: 32,  w: 140, h: 80 },
  compressor_zone: { x: 184, y: 32,  w: 164, h: 80 },
  control_room:    { x: 700, y: 32,  w: 130, h: 80 },
};

// Process-flow connections drawn behind the zones, routed as right angles like a real
// plant schematic. Order: feed -> reactor -> separator -> stripper, with the recycle
// loop reactor -> compressor -> condenser -> separator feeding back.
const PIPES = [
  ["feed_zone", "reactor_zone"],
  ["reactor_zone", "separator_zone"],
  ["separator_zone", "stripper_zone"],
  ["reactor_zone", "compressor_zone"],
  ["compressor_zone", "condenser_zone"],
  ["condenser_zone", "separator_zone"],
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
  // offset: leave the side of A facing B, run to a mid-x, then into the top/bottom of B
  const aRight = cb.cx > ca.cx;
  const startX = aRight ? a.x + a.w : a.x;
  const startY = ca.cy;
  const endX = cb.cx;
  const endY = cb.cy < ca.cy ? b.y + b.h : b.y;
  const midX = (startX + endX) / 2;
  return `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`;
}

const PERMIT_SHORT = { hot_work: "HOT WORK", confined_space: "CONFINED", electrical: "ELECTRICAL", general: "GENERAL" };

export default function PlantMap({ zones, riskByZone, baselineByZone, activeZone, permits, workerPresence }) {
  if (!zones) {
    return (
      <div style={{ height: 360, display: "flex", alignItems: "center", justifyContent: "center", color: "#565f73", fontSize: 13 }}>
        Loading plant layout…
      </div>
    );
  }
  const width = 860;
  const height = 360;

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
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="plant-map" preserveAspectRatio="xMidYMid meet">
      <defs>
        <pattern id="permit-hatch" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
          <rect width="7" height="7" fill="none" />
          <line x1="0" y1="0" x2="0" y2="7" stroke="rgba(255,255,255,0.35)" strokeWidth="1.6" />
        </pattern>
      </defs>

      <g className="pipes">
        {PIPES.map(([a, b]) => {
          const za = LAYOUT[a], zb = LAYOUT[b];
          if (!za || !zb) return null;
          return <path key={`${a}-${b}`} className="zone-pipe" d={orthPath(za, zb)} fill="none" />;
        })}
      </g>

      {Object.keys(zones).filter((zoneId) => LAYOUT[zoneId]).map((zoneId) => {
        const z = LAYOUT[zoneId];
        const label = zones[zoneId].label;
        const score = riskByZone[zoneId];
        const baseline = baselineByZone[zoneId];
        const isActive = zoneId === activeZone;
        const isCritical = isActive && score >= 0.9;
        const permit = activePermitByZone[zoneId];
        const workers = workersByZone[zoneId] || 0;

        return (
          <motion.g
            key={zoneId}
            initial={false}
            style={{ filter: glowFilter(isActive ? score : null) }}
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
    </svg>
  );
}
