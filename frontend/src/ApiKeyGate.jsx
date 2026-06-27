import { useState } from "react";
import { motion } from "motion/react";
import { KeyRound } from "lucide-react";
import { apiFetch, getApiKey, setApiKey } from "./api";

/** Every backend route requires X-API-Key once API_KEY_REQUIRED=true (see
 * server/main.py's require_api_key + db/seed.py, which prints one on first seed).
 * This gate just makes that visible instead of the dashboard silently 401-ing. */
export default function ApiKeyGate({ children }) {
  const [hasKey, setHasKey] = useState(!!getApiKey());
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setChecking(true);
    setError(null);
    const trimmed = input.trim();
    setApiKey(trimmed);
    try {
      const res = await apiFetch("/zones");
      if (!res.ok) throw new Error(res.status === 401 ? "Invalid API key" : `Server error (${res.status})`);
      setHasKey(true);
    } catch (err) {
      setError(String(err.message || err));
      setApiKey("");
    } finally {
      setChecking(false);
    }
  }

  if (hasKey) return children;

  return (
    <div className="app" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
      <motion.form
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
        className="live-controls"
        style={{ width: 420 }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <KeyRound size={18} color="var(--accent-cyan)" />
          <strong style={{ fontSize: 15 }}>API key required</strong>
        </div>
        <p style={{ fontSize: 12.5, color: "var(--text-secondary)", margin: "2px 0 8px" }}>
          Run <code>python -m db.seed</code> on the backend if you don't have one yet --
          it prints a key the first time it's run.
        </p>
        <input
          type="password"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="isi_..."
          autoFocus
          style={{
            background: "var(--bg-panel-raised)", border: "1px solid var(--border-subtle)",
            color: "var(--text-primary)", padding: "10px 12px", borderRadius: "var(--radius-sm)",
            fontFamily: "var(--font-mono)", fontSize: 13, width: "100%",
          }}
        />
        {error && <div className="status-banner error" style={{ marginTop: 10 }}>{error}</div>}
        <button type="submit" className="replay-btn" disabled={checking || !input.trim()} style={{ marginTop: 12, width: "100%" }}>
          {checking ? "Checking..." : "Continue"}
        </button>
      </motion.form>
    </div>
  );
}
