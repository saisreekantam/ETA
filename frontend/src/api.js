export const API_BASE = "http://localhost:8000";

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
