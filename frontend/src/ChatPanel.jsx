import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { BookOpen, MessageSquare, Send, Sparkles, X } from "lucide-react";
import { sendChat } from "./api";

// Prompts that show what the assistant can actually do (plant state + regulatory
// corpus) -- clicking one sends it, so the demo never starts from a blank box.
const SUGGESTIONS = [
  "Why is reactor_zone flagged?",
  "Run the reactor heat-removal scenario",
  "How do I add my own IoT sensor?",
  "What does the Factories Act require before hot work?",
];

const ACTION_LABELS = {
  navigate: (a) => `Opening ${a.args.page}`,
  run_scenario: (a) => `Running scenario ${String(a.args.run_id).slice(0, 6)}`,
  start_replay: (a) => `Opening replay for ${String(a.args.run_id).slice(0, 6)}`,
};

export default function ChatPanel({ open, onClose, onAction }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const threadRef = useRef(null);

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  async function ask(question) {
    const q = question.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: q }]);
    try {
      const res = await sendChat(q, history);
      setMessages((m) => [...m, { role: "assistant", content: res.answer, citations: res.citations, actions: res.actions }]);
      // the copilot's chosen UI actions -- executed by App (navigate, run, replay)
      for (const action of res.actions || []) onAction?.(action);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: String(e), error: true }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          className="chat-panel"
          initial={{ x: 380, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 380, opacity: 0 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          aria-label="Operator chat"
        >
          <div className="chat-head">
            <span className="chat-title">
              <MessageSquare size={14} /> Operator chat
              <span className="tag-muted">local LLM · RAG-grounded</span>
            </span>
            <button className="icon-btn" onClick={onClose} aria-label="Close chat"><X size={15} /></button>
          </div>

          <div className="chat-thread" ref={threadRef}>
            {messages.length === 0 && (
              <div className="chat-empty">
                <p>Ask about the current plant state, safety procedures, or incident history.
                  Answers cite the regulatory corpus — never invented section numbers.</p>
                <div className="chat-suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} className="chat-chip" onClick={() => ask(s)} disabled={busy}>{s}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`chat-msg ${m.role}${m.error ? " error" : ""}`}>
                {m.actions?.length > 0 && (
                  <div className="chat-actions">
                    {m.actions.map((a, j) => (
                      <span key={j} className="chat-action-chip">
                        <Sparkles size={11} /> {ACTION_LABELS[a.tool] ? ACTION_LABELS[a.tool](a) : a.tool}
                      </span>
                    ))}
                  </div>
                )}
                <div className="chat-bubble">{m.content}</div>
                {m.citations && m.citations.length > 0 && (
                  <details className="chat-citations">
                    <summary><BookOpen size={11} /> {m.citations.length} regulatory source(s)</summary>
                    <ul>
                      {m.citations.map((c, j) => (
                        <li key={j}>
                          <strong>{c.source}</strong>
                          <span className="chat-citation-text">{c.text.slice(0, 180)}…</span>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            ))}
            {busy && (
              <div className="chat-msg assistant">
                <div className="chat-bubble chat-typing"><span /><span /><span /></div>
              </div>
            )}
          </div>

          <form
            className="chat-input-row"
            onSubmit={(e) => { e.preventDefault(); ask(input); }}
          >
            <input
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about plant state or safety procedures…"
              disabled={busy}
            />
            <button type="submit" className="chat-send" disabled={busy || !input.trim()} aria-label="Send">
              <Send size={14} />
            </button>
          </form>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
