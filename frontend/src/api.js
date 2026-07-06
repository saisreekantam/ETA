// Defaults to the locally-published backend port; override at build time with
// VITE_API_BASE (e.g. in Docker) without touching code.
export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const API_KEY_STORAGE = "isi_api_key";

export function getApiKey() {
  return localStorage.getItem(API_KEY_STORAGE) || "";
}

export function setApiKey(key) {
  localStorage.setItem(API_KEY_STORAGE, key);
}

/** Wraps fetch with the X-API-Key header -- use this everywhere instead of raw fetch
 * (LiveMonitoring.jsx/Replay.jsx do their own polling fetches and also import this). */
export async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}), "X-API-Key": getApiKey() };
  return fetch(`${API_BASE}${path}`, { ...options, headers });
}

export async function getZones() {
  const res = await apiFetch("/zones");
  return res.json();
}

export async function getScenarios() {
  const res = await apiFetch("/scenarios");
  return res.json();
}

export async function runScenario(runId, { force = false } = {}) {
  const res = await apiFetch(`/run/${runId}${force ? "?force=true" : ""}`, { method: "POST" });
  if (!res.ok) throw new Error(`Run failed: ${res.status}${res.status === 401 ? " (check API key)" : ""}`);
  return res.json();
}

export async function getReplay(runId, step = 4) {
  const res = await apiFetch(`/replay/${runId}?step=${step}`);
  if (!res.ok) throw new Error(`Replay failed: ${res.status}${res.status === 401 ? " (check API key)" : ""}`);
  return res.json();
}

// --- devices / IoT ---------------------------------------------------------------

export async function getDevices() {
  const res = await apiFetch("/devices");
  if (!res.ok) throw new Error(`Devices failed: ${res.status}`);
  return res.json();
}

export async function createDevice(body) {
  const res = await apiFetch("/devices", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail || `Create failed: ${res.status}`);
  return res.json();
}

export async function deleteDevice(id) {
  const res = await apiFetch(`/devices/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
  return res.json();
}

export async function testDevice(id) {
  const res = await apiFetch(`/devices/${id}/test`, { method: "POST" });
  if (!res.ok) throw new Error(`Test failed: ${res.status}`);
  return res.json();
}

export async function getDeviceReadings(id, limit = 40) {
  const res = await apiFetch(`/devices/${id}/readings?limit=${limit}`);
  if (!res.ok) throw new Error(`Readings failed: ${res.status}`);
  return res.json();
}

// --- live / benchmarks / alerts ----------------------------------------------------

/** EventSource can't set headers, so the key rides as a query param (same pattern as
 * the MJPEG stream). */
export function liveStreamUrl(runId, interval = 1.2) {
  const key = encodeURIComponent(getApiKey());
  return `${API_BASE}/live/risk-stream?run_id=${runId}&interval=${interval}&api_key=${key}`;
}

export async function getBenchmarks() {
  const res = await apiFetch("/benchmarks");
  if (!res.ok) throw new Error(`Benchmarks failed: ${res.status}`);
  return res.json();
}

export async function getAlerts({ unackedOnly = false } = {}) {
  const res = await apiFetch(`/alerts${unackedOnly ? "?unacked_only=true" : ""}`);
  if (!res.ok) throw new Error(`Alerts failed: ${res.status}`);
  return res.json();
}

export async function ackAlert(id, name) {
  const res = await apiFetch(`/alerts/${id}/ack`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ acknowledged_by: name }),
  });
  if (!res.ok) throw new Error(`Ack failed: ${res.status}`);
  return res.json();
}

export function evidenceUrl(runId) {
  return `${API_BASE}/runs/${runId}/evidence?api_key=${encodeURIComponent(getApiKey())}`;
}
