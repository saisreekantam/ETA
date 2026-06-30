import { motion } from "motion/react";
import {
  ShieldAlert, Network, FileCheck2, ScrollText, Eye, ServerCog,
} from "lucide-react";

const CAPABILITIES = [
  {
    icon: Network,
    title: "Compound-risk detection",
    body: "A heterogeneous graph neural network scores every plant zone for co-occurring hazards — the failure mode a single sensor threshold never flags on its own.",
  },
  {
    icon: FileCheck2,
    title: "Permit correlation",
    body: "A rule-based agent cross-references each high-risk zone against active work permits, surfacing exactly when authorized work coincides with a hazardous condition.",
  },
  {
    icon: Eye,
    title: "Live CCTV detection",
    body: "Four fine-tuned RT-DETR models run on real video — PPE compliance, unauthorized zone entry, fall/man-down, and fire/smoke — over the same code path for uploaded clips or a live feed.",
  },
  {
    icon: ScrollText,
    title: "Regulatory-grounded reports",
    body: "Retrieval over a DGMS / OISD / Factory Act corpus grounds every generated incident report in real regulatory text, cited verbatim rather than recalled.",
  },
  {
    icon: ServerCog,
    title: "On-prem, auditable",
    body: "The report LLM runs locally via Ollama — no SCADA or safety data leaves the plant network — and every agent decision is checkpointed for a full audit trail.",
  },
];

export default function About() {
  return (
    <div className="about">
      <div className="about-hero">
        <div className="header-icon about-hero-icon">
          <ShieldAlert size={26} color="var(--accent-cyan)" strokeWidth={2.2} />
        </div>
        <h2>Industrial Safety Intelligence</h2>
        <p className="about-lede">
          An intelligence layer for industrial plants that detects compound risk — a
          hazardous condition co-occurring with normal operating activity that no single
          sensor would flag alone — correlates it against work permits and live CCTV, and
          drafts a regulation-grounded incident report, all on-premise.
        </p>
      </div>

      <div className="about-grid">
        {CAPABILITIES.map(({ icon: Icon, title, body }, i) => (
          <motion.div
            key={title}
            className="about-card"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06, duration: 0.3 }}
          >
            <div className="about-card-icon"><Icon size={18} /></div>
            <h3>{title}</h3>
            <p>{body}</p>
          </motion.div>
        ))}
      </div>

      <div className="about-context">
        <h3>Why it matters</h3>
        <p>
          The compound-risk failure mode is exactly what drove incidents like the
          Visakhapatnam Steel Plant coke-oven explosion (Jan 2025) and the Sigachi
          Industries dust explosion (Jun 2025): a hazardous condition and a routine
          activity co-occurring, with no layer fusing the two before it was too late.
          India records 6,500+ fatal workplace accidents a year in its heavy industrial
          sector — this system targets the gap between a signal existing and someone
          acting on it.
        </p>
      </div>
    </div>
  );
}
