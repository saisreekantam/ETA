import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Film, Upload, Webcam, Play, Square, AlertCircle, Flame, VideoOff } from "lucide-react";
import { API_BASE, apiFetch, getApiKey } from "./api";

const SOURCE_MODES = [
  { key: "sample", label: "Sample clip (demo)", icon: Film },
  { key: "upload", label: "Upload video (demo)", icon: Upload },
  { key: "live", label: "Live camera (production)", icon: Webcam },
];

/** Detection events repeat on every frame, so a raw list is dozens of identical
 * rows. Collapse ALL identical events (same detector + event + detail) into one
 * row carrying a total count and the time it was last seen, ordered by most recent
 * -- this is how real monitoring consoles surface "still happening" without a wall
 * of duplicates. Deduping across the whole list (not just consecutive) matters here
 * because the detectors interleave frame-to-frame. */
function groupEvents(events) {
  const byKey = new Map();
  for (const e of events) {
    const key = `${e.detector}|${e.event}|${e.detail}`;
    const g = byKey.get(key);
    if (g) {
      g.count += 1;
      g.lastTs = e.timestamp;
    } else {
      byKey.set(key, { ...e, count: 1, lastTs: e.timestamp });
    }
  }
  return [...byKey.values()].sort((a, b) => (b.lastTs ?? 0) - (a.lastTs ?? 0));
}

function fmtTime(ts) {
  if (ts == null) return "";
  const d = typeof ts === "number" ? new Date(ts * (ts < 1e12 ? 1000 : 1)) : new Date(ts);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString([], { hour12: false });
}

export default function LiveMonitoring({ zones }) {
  const [sourceMode, setSourceMode] = useState("sample"); // "sample" | "upload" | "live"
  const [zone, setZone] = useState("reactor_zone");
  const [runIntrusion, setRunIntrusion] = useState(true);
  const [runFall, setRunFall] = useState(true);
  const [runFireSmoke, setRunFireSmoke] = useState(true);
  const [hasPermit, setHasPermit] = useState(false);
  const [uploadedPath, setUploadedPath] = useState(null);
  const [session, setSession] = useState(null);
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => () => stopPolling(), []);

  function stopPolling() {
    if (pollRef.current) clearInterval(pollRef.current);
  }

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const res = await apiFetch("/vision/upload", { method: "POST", body: form });
    const data = await res.json();
    setUploadedPath(data.path);
  }

  async function startSession() {
    setError(null);
    const params = new URLSearchParams({
      zone, run_intrusion: String(runIntrusion), run_fall: String(runFall),
      run_fire_smoke: String(runFireSmoke), has_active_permit: String(hasPermit),
    });
    if (sourceMode === "live") {
      params.set("live", "true");
      params.set("camera_index", "0");
    } else {
      params.set("live", "false");
      params.set("video_path", sourceMode === "upload" ? uploadedPath : "data/ppe_vision/cached_outputs/sample_demo_clip.mp4");
    }
    try {
      const res = await apiFetch(`/vision/sessions?${params.toString()}`, { method: "POST" });
      if (!res.ok) throw new Error(`Failed to start session: ${res.status}`);
      const data = await res.json();
      setSession(data);
      pollRef.current = setInterval(async () => {
        const evRes = await apiFetch(`/vision/sessions/${data.session_id}/events`);
        const evData = await evRes.json();
        setStatus(evData);
        setEvents(evData.events);
        if (evData.error) setError(evData.error);
      }, 1000);
    } catch (e) {
      setError(String(e));
    }
  }

  async function stopSession() {
    if (session) {
      await apiFetch(`/vision/sessions/${session.session_id}/stop`, { method: "POST" });
    }
    stopPolling();
    setSession(null);
    setStatus(null);
  }

  const grouped = groupEvents(events);
  // Only show the live frame when the session is genuinely streaming -- a failed camera
  // open still creates a session, so gating on `session` alone shows a fake LIVE feed
  // over an empty frame next to the error. status.error or status.running === false
  // means the feed never came up.
  const isStreaming = !!session && !error && status?.running !== false;

  return (
    <div className="live-monitoring">
      <div className="live-controls">
        <div className="live-control-group">
          <label>Source</label>
          <div className="mode-toggle">
            {SOURCE_MODES.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                className={sourceMode === key ? "mode-btn active" : "mode-btn"}
                onClick={() => setSourceMode(key)} disabled={!!session}
              >
                <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                  <Icon size={13} /> {label}
                </span>
              </button>
            ))}
          </div>
        </div>

        <AnimatePresence>
          {sourceMode === "upload" && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="live-control-group">
              <input type="file" accept="video/*" onChange={handleUpload} disabled={!!session} />
              {uploadedPath && <span className="upload-ok">uploaded ✓</span>}
            </motion.div>
          )}

          {sourceMode === "live" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="status-banner loading">
              Opens the server machine's camera (device 0). In production this points at an
              RTSP plant CCTV feed instead -- same code path, different source. Requires OS
              camera permission to be granted to the process running the backend.
            </motion.div>
          )}
        </AnimatePresence>

        <div className="live-control-group">
          <label>Zone this camera covers</label>
          <select value={zone} onChange={(e) => setZone(e.target.value)} disabled={!!session}>
            {zones && Object.keys(zones).map((z) => <option key={z} value={z}>{z}</option>)}
          </select>
        </div>

        <div className="live-control-group checkboxes">
          <label><input type="checkbox" checked={runIntrusion} onChange={(e) => setRunIntrusion(e.target.checked)} disabled={!!session} /> Run zone-intrusion check</label>
          <label><input type="checkbox" checked={runFall} onChange={(e) => setRunFall(e.target.checked)} disabled={!!session} /> Run fall/man-down check</label>
          <label><input type="checkbox" checked={runFireSmoke} onChange={(e) => setRunFireSmoke(e.target.checked)} disabled={!!session} /> <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><Flame size={12} /> Run fire/smoke check</span></label>
          <label><input type="checkbox" checked={hasPermit} onChange={(e) => setHasPermit(e.target.checked)} disabled={!!session} /> Active permit covers this zone</label>
        </div>

        {!session ? (
          <button className="replay-btn" onClick={startSession} disabled={sourceMode === "upload" && !uploadedPath}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><Play size={13} /> Start detection</span>
          </button>
        ) : (
          <button className="replay-btn stop" onClick={stopSession}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><Square size={12} /> Stop</span>
          </button>
        )}
        <AnimatePresence>
          {error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="status-banner error">
              <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><AlertCircle size={14} /> {error}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {session && (
          <motion.div
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="live-feed"
          >
            <div className="live-stream-frame">
              {isStreaming ? (
                <>
                  <img
                    src={`${API_BASE}/vision/sessions/${session.session_id}/stream?api_key=${encodeURIComponent(getApiKey())}`}
                    alt="live detection feed"
                    className="live-stream-image"
                  />
                  <span className="live-rec-badge">
                    <motion.span className="live-rec-dot" animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
                    LIVE
                  </span>
                </>
              ) : (
                <div className="live-stream-unavailable">
                  <VideoOff size={28} />
                  <span>{error ? "Feed unavailable" : "Connecting to feed…"}</span>
                </div>
              )}
            </div>
            <div className="live-side">
              <div className="live-status">
                {status && <>Frames processed: {status.frames_processed} · {status.running ? "running" : "stopped"}</>}
              </div>
              <h3>
                Detection events
                {grouped.length > 0 && <span className="live-event-count">{grouped.length} active</span>}
              </h3>
              {grouped.length === 0 ? (
                <div className="live-events-empty">No detections yet — monitoring the feed…</div>
              ) : (
                <ul className="live-events">
                  <AnimatePresence initial={false}>
                    {grouped.map((e) => (
                      <motion.li
                        key={`${e.detector}-${e.event}-${e.detail}`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.25 }}
                        className={`live-event ${e.event}`}
                      >
                        <div className="live-event-head">
                          <span className="live-event-detector">{e.detector}</span>
                          {e.count > 1 && <span className="live-event-badge">×{e.count}</span>}
                          {fmtTime(e.lastTs) && <span className="live-event-time">{fmtTime(e.lastTs)}</span>}
                        </div>
                        <div className="live-event-detail">{e.detail}</div>
                      </motion.li>
                    ))}
                  </AnimatePresence>
                </ul>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
