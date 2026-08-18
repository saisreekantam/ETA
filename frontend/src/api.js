// Defaults to the locally-published backend port; override at build time with
// VITE_API_BASE (e.g. in Docker) without touching code.
export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const API_KEY_STORAGE = "isi_api_key";
const FACILITY_STORAGE = "isi_facility";

export function getApiKey() {
  return localStorage.getItem(API_KEY_STORAGE) || "";
}

export function setApiKey(key) {
  localStorage.setItem(API_KEY_STORAGE, key);
}

/** The facility chosen at login ({id, name, industry_type, is_demo}) -- every request
 * is scoped to it (see apiFetch). Cleared by the appbar's facility switcher. */
export function getFacility() {
  try {
    return JSON.parse(localStorage.getItem(FACILITY_STORAGE)) || null;
  } catch {
    return null;
  }
}

export function setFacility(facility) {
  if (facility) localStorage.setItem(FACILITY_STORAGE, JSON.stringify(facility));
  else localStorage.removeItem(FACILITY_STORAGE);
}

/** Wraps fetch with the X-API-Key header and scopes every request to the logged-in
 * facility via the facility_id query param all backend routers accept (endpoints that
 * don't take it ignore the extra param). Use this everywhere instead of raw fetch
 * (LiveMonitoring.jsx/Replay.jsx do their own polling fetches and also import this). */
export async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}), "X-API-Key": getApiKey() };
  const facility = getFacility();
  if (facility?.id) {
    path += (path.includes("?") ? "&" : "?") + `facility_id=${encodeURIComponent(facility.id)}`;
  }
  return fetch(`${API_BASE}${path}`, { ...options, headers });
}

// --- facilities / onboarding -------------------------------------------------------

export async function getFacilities() {
  const res = await apiFetch("/facilities");
  if (!res.ok) throw new Error(`Facilities failed: ${res.status}`);
  return res.json();
}

export async function getFacilityTemplates() {
  const res = await apiFetch("/facility-templates");
  if (!res.ok) throw new Error(`Templates failed: ${res.status}`);
  return res.json();
}

export async function createFacility(body) {
  const res = await apiFetch("/facilities", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail || `Create failed: ${res.status}`);
  return res.json();
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

export async function getGeneralizationBenchmarks() {
  const res = await apiFetch("/benchmarks/generalization");
  if (!res.ok) throw new Error(`Generalization benchmarks failed: ${res.status}`);
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

/** Real, time-scoped permit lookup for a zone -- replaces LiveMonitoring's old
 * self-reported "has permit" checkbox with what the server actually finds. */
export async function getActivePermits(zoneKey) {
  const res = await apiFetch(`/zones/${encodeURIComponent(zoneKey)}/active-permits`);
  if (!res.ok) throw new Error(`Active permits failed: ${res.status}`);
  return res.json();
}

export async function getBreakMode() {
  const res = await apiFetch("/break-mode");
  if (!res.ok) throw new Error(`Break mode fetch failed: ${res.status}`);
  return res.json();
}

export async function setBreakMode(active, operatorName) {
  const res = await apiFetch("/break-mode", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active, operator_name: operatorName || null }),
  });
  if (!res.ok) throw new Error(`Break mode update failed: ${res.status}`);
  return res.json();
}

/** Operator chat: question + rolling history (client-held; the endpoint is stateless).
 * History entries are {role, content} only -- citations stay client-side. */
export async function sendChat(question, history = []) {
  const res = await apiFetch("/chat", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history, facility_id: getFacility()?.id || null }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}${res.status === 401 ? " (check API key)" : ""}`);
  return res.json();
}

export function evidenceUrl(runId) {
  return `${API_BASE}/runs/${runId}/evidence?api_key=${encodeURIComponent(getApiKey())}`;
}

/** Every alert this facility has raised, as a CSV download -- for offline analysis (see
 * server/alerts.py::export_alerts_csv). Plain link href, not apiFetch, same reasoning as
 * evidenceUrl: the browser navigates to this directly, so the key rides as a query param. */
export function alertsExportUrl() {
  const facility = getFacility();
  const params = new URLSearchParams({ api_key: getApiKey() });
  if (facility?.id) params.set("facility_id", facility.id);
  return `${API_BASE}/alerts/export?${params.toString()}`;
}
