import { motion } from "motion/react";

function riskColor(score) {
  if (score == null) return "#222b3a";
  if (score >= 0.9) return "#d96363";
  if (score >= 0.7) return "#d9824a";
  if (score >= 0.5) return "#d9a441";
  return "#3f8f72";
}

function glowFilter(score) {
  if (score == null || score < 0.5) return "none";
  const color = riskColor(score);
  return `drop-shadow(0 0 ${6 + score * 14}px ${color}99)`;
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

export default function PlantMap({ zones, riskByZone, baselineByZone, activeZone }) {
  if (!zones) {
    return (
      <div style={{ height: 360, display: "flex", alignItems: "center", justifyContent: "center", color: "#565f73", fontSize: 13 }}>
        Loading plant layout…
      </div>
    );
  }
  const width = 860;
  const height = 360;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="plant-map" preserveAspectRatio="xMidYMid meet">
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
                stroke: isActive ? "#ffffff" : "rgba(255,255,255,0.12)",
                strokeWidth: isActive ? 2.5 : 1.5,
                opacity: score == null ? 0.5 : 0.92,
                scale: isActive ? 1.015 : 1,
              }}
              style={{ transformOrigin: `${z.x + z.w / 2}px ${z.y + z.h / 2}px` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            />
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
          </motion.g>
        );
      })}
    </svg>
  );
}
