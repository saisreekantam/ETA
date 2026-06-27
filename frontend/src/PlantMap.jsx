import { motion } from "motion/react";

function riskColor(score) {
  if (score == null) return "#222b3a";
  if (score >= 0.9) return "#fb5858";
  if (score >= 0.7) return "#fb923c";
  if (score >= 0.5) return "#fbbf24";
  return "#2fae6e";
}

function glowFilter(score) {
  if (score == null || score < 0.5) return "none";
  const color = riskColor(score);
  return `drop-shadow(0 0 ${6 + score * 14}px ${color}99)`;
}

export default function PlantMap({ zones, riskByZone, baselineByZone, activeZone }) {
  if (!zones) {
    return (
      <div style={{ height: 360, display: "flex", alignItems: "center", justifyContent: "center", color: "#565f73", fontSize: 13 }}>
        Loading plant layout…
      </div>
    );
  }
  const width = 820;
  const height = 360;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="plant-map">
      {Object.entries(zones).map(([zoneId, z]) => {
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
            <text x={z.x + z.w / 2} y={z.y + z.h / 2 - 6} textAnchor="middle" className="zone-label">
              {z.label}
            </text>
            {score != null && (
              <text x={z.x + z.w / 2} y={z.y + z.h / 2 + 14} textAnchor="middle" className="zone-score">
                G:{score.toFixed(2)} {baseline != null ? `B:${baseline.toFixed(2)}` : ""}
              </text>
            )}
          </motion.g>
        );
      })}
    </svg>
  );
}
