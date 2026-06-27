import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  ShieldAlert, Radio, PlayCircle, Camera, ScrollText, AlertTriangle,
  FileWarning, Quote, Eye, CheckCircle2,
} from "lucide-react";
import { getZones, getScenarios, runScenario } from "./api";
import PlantMap from "./PlantMap";
import Replay from "./Replay";
import LiveMonitoring from "./LiveMonitoring";
import "./App.css";

const ESCALATION_LABEL = {
  none: "Normal",
  monitor: "Monitor",
  alert: "Alert",
  emergency: "EMERGENCY",
};

const MODES = [
  { key: "single", label: "Single run", icon: PlayCircle },
  { key: "replay", label: "Time replay", icon: Radio },
  { key: "live", label: "Live CCTV", icon: Camera },
];

const fadeUp = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -6 },
  transition: { duration: 0.25, ease: "easeOut" },
};

export default function App() {
  const [zones, setZones] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("single"); // "single" | "replay" | "live"
  const [replayFrame, setReplayFrame] = useState(null);

  useEffect(() => {
    getZones().then(setZones);
    getScenarios().then((s) => {
      setScenarios(s);
      if (s.length) setSelectedRunId(s[0].compound_run_ids[0]);
    });
  }, []);

  async function handleRun(runId) {
    setSelectedRunId(runId);
    setReplayFrame(null);
    if (mode !== "single") return; // Replay component drives itself off selectedRunId
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runScenario(runId);
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const handleReplayFrame = useCallback((frame) => setReplayFrame(frame), []);

  const riskByZone = {};
  const baselineByZone = {};
  let activeZone = null;

  if (mode === "replay" && replayFrame) {
    for (const [zone, v] of Object.entries(replayFrame.zones)) {
      riskByZone[zone] = v.gnn;
      baselineByZone[zone] = v.baselineAlert ? 1.0 : 0.0;
    }
    activeZone = replayFrame.trueZone;
  } else if (mode === "single" && result) {
    for (const z of result.zone_risk_scores) {
      riskByZone[z.zone] = z.compound_risk_score;
      baselineByZone[z.zone] = z.baseline_risk_score;
      activeZone = z.zone;
    }
  }

  return (
    <div className="app">
      <motion.header initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="header-icon">
          <ShieldAlert size={20} color="#06141a" strokeWidth={2.4} />
        </div>
        <div>
          <h1>Industrial Safety Intelligence</h1>
          <p className="subtitle">Compound risk detection · permit correlation · regulatory-grounded incident reports</p>
        </div>
      </motion.header>

      <div className="layout">
        <aside className="sidebar">
          <div className="mode-toggle mode-toggle-3">
            {MODES.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                className={mode === key ? "mode-btn active" : "mode-btn"}
                onClick={() => {
                  setMode(key);
                  if (key !== "single") setResult(null);
                  if (key !== "replay") setReplayFrame(null);
                }}
              >
                <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                  <Icon size={13} /> {label}
                </span>
              </button>
            ))}
          </div>

          <AnimatePresence>
            {mode !== "live" && (
              <motion.div {...fadeUp}>
                <h2>Scenarios</h2>
                {scenarios.map((s) => (
                  <div key={s.scenario_id} className="scenario-group">
                    <div className="scenario-name">{s.scenario_id.replace(/_/g, " ")}</div>
                    <div className="scenario-buttons">
                      {s.compound_run_ids.map((rid) => (
                        <button
                          key={rid}
                          className={rid === selectedRunId ? "run-btn active compound" : "run-btn compound"}
                          onClick={() => handleRun(rid)}
                          disabled={loading}
                        >
                          compound {rid.slice(0, 6)}
                        </button>
                      ))}
                      {s.normal_run_ids.map((rid) => (
                        <button
                          key={rid}
                          className={rid === selectedRunId ? "run-btn active normal" : "run-btn normal"}
                          onClick={() => handleRun(rid)}
                          disabled={loading}
                        >
                          normal {rid.slice(0, 6)}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </aside>

        <main className="main">
          <AnimatePresence mode="wait">
            {mode === "live" ? (
              <motion.div key="live" {...fadeUp}>
                <LiveMonitoring zones={zones} />
              </motion.div>
            ) : (
              <motion.div key="map" {...fadeUp}>
                <div className="plant-map-card">
                  <PlantMap zones={zones} riskByZone={riskByZone} baselineByZone={baselineByZone} activeZone={activeZone} />
                </div>

                {mode === "replay" && selectedRunId && (
                  <Replay runId={selectedRunId} onFrame={handleReplayFrame} />
                )}

                <AnimatePresence>
                  {loading && (
                    <motion.div {...fadeUp} className="status-banner loading">
                      Running pipeline (compound-risk → permit correlation → orchestrator)...
                    </motion.div>
                  )}
                  {error && (
                    <motion.div {...fadeUp} className="status-banner error">{error}</motion.div>
                  )}
                </AnimatePresence>

                <AnimatePresence>
                  {mode === "single" && result && (
                    <motion.div
                      className="result-panel"
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35, ease: "easeOut" }}
                    >
                      <motion.div
                        className={`escalation-badge esc-${result.escalation_level}`}
                        animate={result.escalation_level === "emergency" ? {
                          boxShadow: ["0 0 18px rgba(251,88,88,0.3)", "0 0 38px rgba(251,88,88,0.65)", "0 0 18px rgba(251,88,88,0.3)"],
                        } : {}}
                        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
                      >
                        {result.escalation_level === "emergency" ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
                        {ESCALATION_LABEL[result.escalation_level]}
                      </motion.div>

                      <section>
                        <h3><ScrollText size={13} /> Audit trail</h3>
                        <ol className="audit-log">
                          {result.audit_log.map((entry, i) => (
                            <motion.li
                              key={i}
                              initial={{ opacity: 0, x: -8 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.12, duration: 0.3 }}
                            >
                              <span className="step-num">{i + 1}</span>
                              <span>{entry}</span>
                            </motion.li>
                          ))}
                        </ol>
                      </section>

                      {result.permit_violations.length > 0 && (
                        <section>
                          <h3><FileWarning size={13} /> Permit violations</h3>
                          {result.permit_violations.map((v, i) => (
                            <motion.div
                              key={i}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 0.3 + i * 0.1 }}
                              className={`violation sev-${v.severity}`}
                            >
                              <strong>[{v.severity}]</strong> {v.reason}
                            </motion.div>
                          ))}
                        </section>
                      )}

                      {result.vision_detections && result.vision_detections.length > 0 && (
                        <section>
                          <h3><Eye size={13} /> CCTV / PPE vision evidence (cached, not live)</h3>
                          {result.vision_detections.map((d, i) => (
                            <div key={i} className="vision-card">
                              {d.image_url && (
                                <img src={`http://localhost:8000${d.image_url}`} alt={d.frame_id} className="vision-image" />
                              )}
                              <div className="vision-meta">
                                <div>zone: {d.zone}</div>
                                <div>detections: {d.detections.join(", ") || "none"}</div>
                              </div>
                            </div>
                          ))}
                        </section>
                      )}

                      {result.incident_report && (
                        <section>
                          <h3><Quote size={13} /> Generated incident report (local LLM, no cloud API)</h3>
                          <pre className="incident-report">{result.incident_report}</pre>
                          <div className="citations">
                            <strong>Citations:</strong>
                            <ul>{result.retrieved_citations.map((c, i) => <li key={i}>{c}</li>)}</ul>
                          </div>
                        </section>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
