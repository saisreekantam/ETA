import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import {
  Camera, Cpu, Plus, RefreshCw, Trash2, Video, Wifi, WifiOff, CircleHelp,
} from "lucide-react";
import { API_BASE, createDevice, deleteDevice, getDeviceReadings, getDevices, testDevice } from "./api";

const POLL_MS = 6000;

function StatusDot({ status }) {
  const cls = status === "online" ? "dev-dot online" : status === "offline" ? "dev-dot offline" : "dev-dot";
  return <span className={cls} title={status} />;
}

function timeAgo(iso) {
  if (!iso) return "never";
  const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

/** Minimal single-series sparkline: recessive, 2px line, no axes -- the card's latest
 * value is the readable number; this shows shape only. */
function Sparkline({ points }) {
  if (!points || points.length < 2) return <div className="spark-empty">collecting…</div>;
  const w = 220, h = 44, pad = 3;
  const vals = points.map((p) => p.value);
  const min = Math.min(...vals), max = Math.max(...vals);
  const span = max - min || 1;
  const step = (w - pad * 2) / (points.length - 1);
  const d = vals.map((v, i) =>
    `${i === 0 ? "M" : "L"} ${(pad + i * step).toFixed(1)} ${(h - pad - ((v - min) / span) * (h - pad * 2)).toFixed(1)}`
  ).join(" ");
  return (
    <svg className="sparkline" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <path d={d} fill="none" stroke="var(--accent-cyan)" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CameraCard({ device, onDelete }) {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState(null);

  async function runTest() {
    setTesting(true);
    setResult(null);
    try {
      setResult(await testDevice(device.id));
    } catch (e) {
      setResult({ ok: false, detail: String(e.message || e) });
    } finally {
      setTesting(false);
    }
  }

  return (
    <motion.div className="device-card" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div className="device-head">
        <div className="device-icon"><Video size={16} /></div>
        <div className="device-title">
          <strong>{device.name}</strong>
          <span className="device-sub">{device.source_type}{device.source ? ` · ${device.source.split("/").pop()}` : ""}</span>
        </div>
        <StatusDot status={device.status} />
      </div>
      <div className="device-meta">
        {device.zone_label && <span className="device-chip">{device.zone_label}</span>}
        <span className="device-lastseen">last seen {timeAgo(device.last_seen)}</span>
      </div>
      {result && (
        result.ok ? (
          <div className="device-test-result">
            <img src={result.snapshot} alt={`${device.name} snapshot`} className="device-snapshot" />
            <span className="device-test-ok">Connected · {result.latency_ms} ms · {result.width}×{result.height}</span>
          </div>
        ) : (
          <div className="device-test-fail"><WifiOff size={13} /> {result.detail}</div>
        )
      )}
      {!result && device.last_error && <div className="device-test-fail"><WifiOff size={13} /> {device.last_error}</div>}
      <div className="device-actions">
        <button className="rerun-btn" onClick={runTest} disabled={testing}>
          <Wifi size={13} /> {testing ? "Testing…" : "Test connection"}
        </button>
        <button className="device-delete" onClick={() => onDelete(device)} title="Remove device">
          <Trash2 size={14} />
        </button>
      </div>
    </motion.div>
  );
}

function SensorCard({ device, onDelete }) {
  const [readings, setReadings] = useState(null);
  const [showHint, setShowHint] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = () => getDeviceReadings(device.id).then((r) => { if (alive) setReadings(r); }).catch(() => {});
    load();
    const t = setInterval(load, POLL_MS);
    return () => { alive = false; clearInterval(t); };
  }, [device.id]);

  const curlHint = `curl -X POST ${API_BASE}/iot/readings -H 'Content-Type: application/json' \\\n  -d '{"device_id":"${device.id}","value":42.5}'`;

  return (
    <motion.div className="device-card" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div className="device-head">
        <div className="device-icon"><Cpu size={16} /></div>
        <div className="device-title">
          <strong>{device.name}</strong>
          <span className="device-sub">{device.metric}{device.source_type === "simulated" ? " · simulated" : " · push"}</span>
        </div>
        <StatusDot status={device.status} />
      </div>
      <div className="device-meta">
        {device.zone_label && <span className="device-chip">{device.zone_label}</span>}
        <span className="device-lastseen">last seen {timeAgo(device.last_seen)}</span>
      </div>
      <div className="sensor-value-row">
        <span className="sensor-value">
          {device.latest_value != null ? device.latest_value.toFixed(1) : "—"}
          <small>{device.unit || ""}</small>
        </span>
        <Sparkline points={readings} />
      </div>
      {showHint && <pre className="device-curl">{curlHint}</pre>}
      <div className="device-actions">
        <button className="rerun-btn" onClick={() => setShowHint((v) => !v)}>
          <CircleHelp size={13} /> {showHint ? "Hide" : "Push readings"}
        </button>
        <button className="device-delete" onClick={() => onDelete(device)} title="Remove device">
          <Trash2 size={14} />
        </button>
      </div>
    </motion.div>
  );
}

const EMPTY_FORM = { name: "", kind: "camera", source_type: "rtsp", source: "", metric: "", unit: "", zone: "" };

export default function Devices({ zones }) {
  const [devices, setDevices] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [formError, setFormError] = useState(null);
  const [saving, setSaving] = useState(false);
  const pollRef = useRef(null);

  const refresh = useCallback(() => {
    getDevices().then(setDevices).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    pollRef.current = setInterval(refresh, POLL_MS);
    return () => clearInterval(pollRef.current);
  }, [refresh]);

  function set(k, v) { setForm((f) => ({ ...f, [k]: v })); }

  async function submit(e) {
    e.preventDefault();
    setSaving(true);
    setFormError(null);
    try {
      const body = { ...form };
      if (body.kind === "sensor" && !["push", "simulated"].includes(body.source_type)) body.source_type = "push";
      await createDevice(body);
      setForm(EMPTY_FORM);
      setShowForm(false);
      refresh();
    } catch (err) {
      setFormError(String(err.message || err));
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(device) {
    if (!window.confirm(`Remove ${device.name}?`)) return;
    await deleteDevice(device.id).catch(() => {});
    refresh();
  }

  const cameras = (devices || []).filter((d) => d.kind === "camera");
  const sensors = (devices || []).filter((d) => d.kind === "sensor");
  const zoneKeys = Object.keys(zones || {});

  return (
    <div className="devices-page">
      <div className="devices-header">
        <div>
          <h2>Connected devices</h2>
          <p className="devices-lede">
            Cameras and IoT sensors registered to this facility. Any device that can POST
            JSON can feed the platform — no vendor gateway required.
          </p>
        </div>
        <button className="replay-btn" onClick={() => setShowForm((v) => !v)}>
          <Plus size={14} /> Add device
        </button>
      </div>

      {showForm && (
        <motion.form className="device-form" onSubmit={submit}
                     initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}>
          <div className="device-form-row">
            <label>Name
              <input value={form.name} onChange={(e) => set("name", e.target.value)}
                     placeholder="e.g. North gate cam" required />
            </label>
            <label>Type
              <select value={form.kind} onChange={(e) => set("kind", e.target.value === "camera" ? "camera" : "sensor")}>
                <option value="camera">Camera</option>
                <option value="sensor">IoT sensor</option>
              </select>
            </label>
            <label>Zone
              <select value={form.zone} onChange={(e) => set("zone", e.target.value)}>
                <option value="">— unassigned —</option>
                {zoneKeys.map((k) => <option key={k} value={k}>{zones[k].label}</option>)}
              </select>
            </label>
          </div>
          {form.kind === "camera" ? (
            <div className="device-form-row">
              <label>Source type
                <select value={form.source_type} onChange={(e) => set("source_type", e.target.value)}>
                  <option value="rtsp">Stream URL (RTSP / HTTP / IP-cam)</option>
                  <option value="webcam">Webcam (device index — needs /dev/video* in the backend container)</option>
                  <option value="file">Video file (demo)</option>
                </select>
              </label>
              <label className="grow">Source
                <input value={form.source} onChange={(e) => set("source", e.target.value)}
                       placeholder={form.source_type === "webcam" ? "0" : form.source_type === "rtsp" ? "rtsp://10.0.0.5:554/stream or http://192.168.1.20:8080/video" : "data/…/clip.mp4"} />
              </label>
            </div>
          ) : (
            <div className="device-form-row">
              <label>Mode
                <select value={form.source_type} onChange={(e) => set("source_type", e.target.value)}>
                  <option value="push">Push (device POSTs readings)</option>
                  <option value="simulated">Simulated (demo)</option>
                </select>
              </label>
              <label>Metric
                <input value={form.metric} onChange={(e) => set("metric", e.target.value)} placeholder="gas_ppm" />
              </label>
              <label>Unit
                <input value={form.unit} onChange={(e) => set("unit", e.target.value)} placeholder="ppm" />
              </label>
            </div>
          )}
          {formError && <div className="status-banner error">{formError}</div>}
          <div className="device-form-actions">
            <button type="submit" className="replay-btn" disabled={saving}>{saving ? "Saving…" : "Register device"}</button>
            <button type="button" className="rerun-btn" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </motion.form>
      )}

      <h3 className="devices-section"><Camera size={14} /> Cameras <span className="sidebar-count">{cameras.length}</span></h3>
      <div className="devices-grid">
        {cameras.map((d) => <CameraCard key={d.id} device={d} onDelete={handleDelete} />)}
        {devices && cameras.length === 0 && <p className="devices-empty">No cameras registered yet.</p>}
      </div>

      <h3 className="devices-section"><Cpu size={14} /> IoT sensors <span className="sidebar-count">{sensors.length}</span></h3>
      <div className="devices-grid">
        {sensors.map((d) => <SensorCard key={d.id} device={d} onDelete={handleDelete} />)}
        {devices && sensors.length === 0 && <p className="devices-empty">No sensors registered yet.</p>}
      </div>

      {!devices && <div className="status-banner loading"><span className="spinner" /> Loading devices…</div>}
      <p className="devices-footnote">
        <RefreshCw size={11} /> Health refreshes every {POLL_MS / 1000}s. Camera tests open the
        actual source and return a real frame — status reflects connectivity, not registration.
      </p>
    </div>
  );
}
