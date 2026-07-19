import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Building2, Factory, PlayCircle, Plus } from "lucide-react";
import { createFacility, getFacilities, getFacilityTemplates } from "./api";

/** Second login stage: choose which facility to enter. The seeded demo plant is the
 * guided-demo path (benchmark scenarios + trained GNN); any other facility is a live
 * shell created from an industry template below -- zones, devices, alerts, and chat
 * work immediately, scoped to that facility. */
export default function FacilityPicker({ onSelect }) {
  const [facilities, setFacilities] = useState(null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const load = (attempt = 0) =>
      getFacilities()
        .then((f) => { if (!cancelled) setFacilities(f); })
        .catch(() => { if (!cancelled && attempt < 10) setTimeout(() => load(attempt + 1), 2000); });
    load();
    return () => { cancelled = true; };
  }, []);

  if (creating) {
    return <CreateFacilityWizard onBack={() => setCreating(false)} onCreated={onSelect} />;
  }

  const demo = facilities?.find((f) => f.is_demo);
  const others = facilities?.filter((f) => !f.is_demo) || [];

  return (
    <div className="login-page">
      <motion.div
        initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="login-card facility-card"
      >
        <div className="header-icon login-brand-icon">
          <Building2 size={24} color="var(--accent-cyan)" strokeWidth={2.2} />
        </div>
        <h1>Choose your facility</h1>
        <p className="login-sub">Enter the guided demo plant, or set up your own facility.</p>

        {facilities === null && <p className="login-hint">Loading facilities…</p>}

        {demo && (
          <button className="facility-option demo" onClick={() => onSelect(demo)}>
            <PlayCircle size={18} />
            <span className="facility-option-body">
              <strong>{demo.name}</strong>
              <span>Guided demo — benchmark scenarios, trained GNN, incident reports</span>
            </span>
          </button>
        )}

        {others.map((f) => (
          <button key={f.id} className="facility-option" onClick={() => onSelect(f)}>
            <Factory size={18} />
            <span className="facility-option-body">
              <strong>{f.name}</strong>
              <span>{f.industry_type ? f.industry_type.replace(/_/g, " ") : "custom"} · {f.n_zones} zones{f.location ? ` · ${f.location}` : ""}</span>
            </span>
          </button>
        ))}

        <button className="facility-option create" onClick={() => setCreating(true)}>
          <Plus size={18} />
          <span className="facility-option-body">
            <strong>Set up a new facility</strong>
            <span>Pick an industry template and customize the zones for your plant</span>
          </span>
        </button>

        {error && <div className="status-banner error">{error}</div>}
      </motion.div>
    </div>
  );
}

function CreateFacilityWizard({ onBack, onCreated }) {
  const [templates, setTemplates] = useState(null);
  const [selected, setSelected] = useState(null); // industry_type key
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [zones, setZones] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getFacilityTemplates().then(setTemplates).catch(() => setError("Could not load templates — is the backend up?"));
  }, []);

  function pickTemplate(tpl) {
    setSelected(tpl.industry_type);
    setZones(tpl.zones.map((z) => ({ ...z })));
  }

  async function handleCreate(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const created = await createFacility({
        name: name.trim(), location: location.trim() || null,
        industry_type: selected, zones,
      });
      onCreated(created);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      <motion.form
        onSubmit={handleCreate}
        initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="login-card facility-card wide"
      >
        <button type="button" className="wizard-back" onClick={onBack}><ArrowLeft size={13} /> Back</button>
        <h1>Set up your facility</h1>

        {!selected && (
          <>
            <p className="login-sub">What kind of plant is this? The template pre-lays-out the monitored zones — you can rename them next.</p>
            {templates === null && !error && <p className="login-hint">Loading templates…</p>}
            <div className="template-grid">
              {(templates || []).map((tpl) => (
                <button type="button" key={tpl.industry_type} className="template-card" onClick={() => pickTemplate(tpl)}>
                  <strong>{tpl.label}</strong>
                  <span>{tpl.description}</span>
                  <span className="template-zones">{tpl.zones.length} zones</span>
                </button>
              ))}
            </div>
          </>
        )}

        {selected && (
          <>
            <label className="login-label">Facility name</label>
            <input className="login-input" value={name} onChange={(e) => setName(e.target.value)}
                   placeholder="e.g. Bhilai Rolling Complex" autoFocus />
            <label className="login-label">Location (optional)</label>
            <input className="login-input" value={location} onChange={(e) => setLocation(e.target.value)}
                   placeholder="e.g. Chhattisgarh, India" />

            <label className="login-label">Monitored zones — rename to match your plant</label>
            <div className="zone-edit-list">
              {zones.map((z, i) => (
                <input
                  key={z.key}
                  className="login-input zone-edit-input"
                  value={z.label}
                  onChange={(e) => setZones(zones.map((zz, j) => (j === i ? { ...zz, label: e.target.value } : zz)))}
                />
              ))}
            </div>

            {error && <div className="status-banner error">{error}</div>}
            <button type="submit" className="replay-btn login-btn" disabled={busy || !name.trim()}>
              {busy ? "Creating…" : "Create facility & enter"}
            </button>
          </>
        )}
      </motion.form>
    </div>
  );
}
