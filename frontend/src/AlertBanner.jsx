import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, Flame, Loader2, ShieldAlert, X } from "lucide-react";

const LEVEL_ICON = { emergency: Flame, alert: ShieldAlert, security: ShieldAlert, monitor: AlertTriangle };

/** The bell dropdown is easy to miss in a live pitch -- this is the surface actually meant
 * to be seen: a large, level-colored banner that slides in for every NEW unacknowledged
 * alert (tracked against `alerts`, which App.jsx already polls), stacked so a burst of
 * simultaneous events doesn't silently drop any. Shows the LLM-generated `reasoning` once
 * the backend has it (server/alerts.py::persist_alert generates it synchronously for the
 * sensor pipeline, asynchronously for CCTV-correlated/device/life-safety alerts -- see
 * reasoning_status).
 *
 * `visible` is DERIVED from `alerts`, not separately-owned state -- a banner disappears
 * ONLY when its alert is no longer in `alerts` (acknowledged, from this banner's own
 * button OR the bell dropdown OR another tab/session -- all converge on the same poll) or
 * when the operator explicitly dismisses it (X, tracked in `dismissedIds`, hides the
 * popup without acknowledging). There is deliberately NO auto-dismiss timer -- an alert
 * silently vanishing on its own after N seconds is indistinguishable from "someone
 * handled it" and is exactly the failure mode this banner exists to prevent. */
export default function AlertBanner({ alerts, onAck }) {
  const [order, setOrder] = useState([]); // alert ids, newest first, in first-seen order
  const [dismissedIds, setDismissedIds] = useState(() => new Set());
  const seenIds = useRef(new Set());

  useEffect(() => {
    const freshIds = alerts.map((a) => a.id).filter((id) => !seenIds.current.has(id));
    if (freshIds.length === 0) return;
    freshIds.forEach((id) => seenIds.current.add(id));
    setOrder((o) => [...freshIds, ...o]);
  }, [alerts]);

  function dismiss(id) {
    setDismissedIds((s) => new Set(s).add(id));
  }

  const visible = order
    .filter((id) => !dismissedIds.has(id))
    .map((id) => alerts.find((a) => a.id === id))
    .filter(Boolean) // no longer in `alerts` -> acknowledged somewhere -> stop showing it
    .slice(0, 4);

  return (
    <div className="alert-banner-stack">
      <AnimatePresence initial={false}>
        {visible.map((a) => {
          const Icon = LEVEL_ICON[a.level] || AlertTriangle;
          return (
            <motion.div
              key={a.id}
              className={`alert-banner level-${a.level}`}
              initial={{ opacity: 0, y: -24, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, x: 40 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
            >
              <Icon size={20} className="alert-banner-icon" />
              <div className="alert-banner-body">
                <div className="alert-banner-head">
                  <span className="alert-banner-level">{a.level}</span>
                  {a.zone_label && a.zone_label !== "Facility-wide" && (
                    <span className="alert-banner-zone">{a.zone_label}</span>
                  )}
                </div>
                <div className="alert-banner-msg">{a.message}</div>
                <div className="alert-banner-reasoning">
                  {a.reasoning_status === "pending" ? (
                    <span className="alert-banner-generating"><Loader2 size={12} className="spin" /> Generating explanation…</span>
                  ) : a.reasoning_status === "ready" && a.reasoning ? (
                    <span>{a.reasoning}</span>
                  ) : null}
                </div>
              </div>
              <div className="alert-banner-actions">
                <button className="rerun-btn" onClick={() => onAck(a)}>Acknowledge</button>
                <button className="alert-banner-close" onClick={() => dismiss(a.id)} aria-label="Dismiss">
                  <X size={14} />
                </button>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
