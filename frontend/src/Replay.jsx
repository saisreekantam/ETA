import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { getReplay } from "./api";

const TICK_MS = 250;

export default function Replay({ runId, onFrame }) {
  const [trace, setTrace] = useState(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    setTrace(null);
    setFrameIdx(0);
    setPlaying(false);
    getReplay(runId).then(setTrace);
  }, [runId]);

  useEffect(() => {
    if (!playing || !trace) return;
    intervalRef.current = setInterval(() => {
      setFrameIdx((i) => {
        if (i >= trace.cutoffs.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, TICK_MS);
    return () => clearInterval(intervalRef.current);
  }, [playing, trace]);

  useEffect(() => {
    if (!trace) return;
    const frame = {};
    for (const zone of Object.keys(trace.trace)) {
      frame[zone] = {
        gnn: trace.trace[zone].gnn[frameIdx],
        baselineAlert: trace.trace[zone].baseline_alert[frameIdx],
      };
    }
    onFrame({
      sample: trace.cutoffs[frameIdx],
      onset: trace.ground_truth_onset_sample,
      trueZone: trace.true_zone,
      zones: frame,
    });
  }, [trace, frameIdx, onFrame]);

  if (!trace) return <div className="replay-bar loading">Loading replay trace...</div>;

  const sample = trace.cutoffs[frameIdx];
  const onsetReached = trace.ground_truth_onset_sample != null && sample >= trace.ground_truth_onset_sample;

  const isDone = frameIdx >= trace.cutoffs.length - 1;

  return (
    <motion.div className="replay-bar" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <button className="replay-btn" onClick={() => setPlaying((p) => !p)}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          {playing ? <Pause size={13} /> : isDone ? <RotateCcw size={13} /> : <Play size={13} />}
          {playing ? "Pause" : isDone ? "Replay" : "Play"}
        </span>
      </button>
      <input
        type="range"
        min={0}
        max={trace.cutoffs.length - 1}
        value={frameIdx}
        onChange={(e) => { setPlaying(false); setFrameIdx(Number(e.target.value)); }}
      />
      <span className="replay-time">
        sample {sample} (t={sample * 3}min){trace.ground_truth_onset_sample != null && (
          <> &middot; onset at sample {trace.ground_truth_onset_sample}{" "}
            <motion.strong
              className={onsetReached ? "onset-reached" : ""}
              animate={onsetReached ? { opacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 1, repeat: Infinity }}
            >
              {onsetReached ? "(reached)" : "(not yet)"}
            </motion.strong>
          </>
        )}
      </span>
    </motion.div>
  );
}
