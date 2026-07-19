import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { KeyRound, ShieldAlert } from "lucide-react";
import { apiFetch, getApiKey, getFacility, setApiKey, setFacility } from "./api";
import FacilityPicker from "./FacilityPicker";

/** The login flow, two stages:
 *  1. API key -- every backend route requires X-API-Key once API_KEY_REQUIRED=true
 *     (see server/main.py's require_api_key + db/seed.py, which prints one on first
 *     seed). Probed on mount: when the backend doesn't require keys (dev), this stage
 *     auto-skips instead of demanding a meaningless key.
 *  2. Facility -- pick the guided demo plant or create/enter a custom facility
 *     (FacilityPicker). Everything the dashboard fetches afterwards is scoped to it. */
export default function ApiKeyGate({ children }) {
  const [stage, setStage] = useState(() =>
    getFacility() ? "app" : getApiKey() ? "facility" : "probing");
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  // Dev convenience: if the backend accepts keyless requests, skip straight to the
  // facility stage. On any failure (backend down, 401) fall back to the key form.
  useEffect(() => {
    if (stage !== "probing") return;
    let cancelled = false;
    apiFetch("/zones")
      .then((res) => { if (!cancelled) setStage(res.ok ? "facility" : "key"); })
      .catch(() => { if (!cancelled) setStage("key"); });
    return () => { cancelled = true; };
  }, [stage]);

  async function handleSubmit(e) {
    e.preventDefault();
    setChecking(true);
    setError(null);
    const trimmed = input.trim();
    setApiKey(trimmed);
    try {
      const res = await apiFetch("/zones");
      if (!res.ok) throw new Error(res.status === 401 ? "Invalid API key" : `Server error (${res.status})`);
      setStage("facility");
    } catch (err) {
      setError(String(err.message || err));
      setApiKey("");
    } finally {
      setChecking(false);
    }
  }

  if (stage === "app") return children;

  if (stage === "facility") {
    return (
      <FacilityPicker
        onSelect={(facility) => {
          setFacility(facility);
          setStage("app");
        }}
      />
    );
  }

  if (stage === "probing") {
    return <div className="login-page"><p className="login-hint">Connecting…</p></div>;
  }

  return (
    <div className="login-page">
      <motion.form
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="login-card"
      >
        <div className="header-icon login-brand-icon">
          <ShieldAlert size={24} color="var(--accent-cyan)" strokeWidth={2.2} />
        </div>
        <h1>Industrial Safety Intelligence</h1>
        <p className="login-sub">Sign in with your facility API key to open the control room.</p>

        <label className="login-label" htmlFor="api-key-input">
          <KeyRound size={13} /> Facility API key
        </label>
        <input
          id="api-key-input"
          type="password"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="isi_..."
          autoFocus
          className="login-input"
        />
        {error && <div className="status-banner error">{error}</div>}
        <button type="submit" className="replay-btn login-btn" disabled={checking || !input.trim()}>
          {checking ? "Checking…" : "Sign in"}
        </button>
        <p className="login-hint">
          No key yet? Run <code>python -m db.seed</code> on the backend — it prints one the
          first time it seeds. Keys are scoped per facility and verified server-side.
        </p>
      </motion.form>
    </div>
  );
}
