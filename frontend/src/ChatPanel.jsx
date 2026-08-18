import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { BookOpen, MessageSquare, Send, Sparkles, Trash2, X } from "lucide-react";
import { getFacility, sendChat } from "./api";

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

// The LLM writes markdown (**bold**, "- " bullets); the bubble previously rendered that
// literally as asterisks. No markdown lib pulled in for this -- just the two constructs
// qwen actually produces in practice, parsed into real React elements (never
// dangerouslySetInnerHTML, so nothing the model writes can inject markup).
function renderBoldSpans(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
      : part
  );
}

function renderChatContent(content) {
  const lines = String(content).split("\n");
  return lines.map((line, i) => {
    const bullet = line.match(/^\s*[-*]\s+(.*)/);
    const body = renderBoldSpans(bullet ? bullet[1] : line, i);
    return (
      <div key={i} className={bullet ? "chat-line chat-line-bullet" : "chat-line"}>
        {bullet && <span className="chat-bullet-dot" aria-hidden="true" />}
        {body}
      </div>
    );
  });
}

// --- short-term memory -------------------------------------------------------------
// The thread is kept in sessionStorage: it survives a reload or a stray navigation
// (the common way an operator loses a conversation mid-task) but is gone when the tab
// closes -- short-term by construction, and no plant Q&A left on a shared control-room
// machine. Scoped per facility so switching plants never carries stale context across.
const MEMORY_PREFIX = "isi_chat_v1";
// The server keeps only the last few turns anyway (MAX_HISTORY_TURNS); this bounds what
// we persist so a long shift can't grow the store without limit.
const MEMORY_MAX_MESSAGES = 40;
// Matches the server's ChatMessage max_length -- an over-long bubble replayed as history
// would fail request validation and break every later turn.
const MAX_CONTENT = 4000;

function memoryKey() {
  return `${MEMORY_PREFIX}:${getFacility()?.id || "default"}`;
}

function loadMemory() {
  try {
    const raw = sessionStorage.getItem(memoryKey());
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return []; // corrupt/unavailable storage must never block the chat
  }
}

function saveMemory(messages) {
  try {
    sessionStorage.setItem(memoryKey(), JSON.stringify(messages.slice(-MEMORY_MAX_MESSAGES)));
  } catch {
    /* private mode / quota -- memory degrades to in-session only */
  }
}

export default function ChatPanel({ open, onClose, onAction }) {
  const [messages, setMessages] = useState(loadMemory);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const threadRef = useRef(null);

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  useEffect(() => { saveMemory(messages); }, [messages]);

  // The persistence effect writes the emptied thread straight back, so clearing state is
  // enough -- no separate storage removal needed.
  function clearMemory() {
    setMessages([]);
  }

  async function ask(question) {
    const q = question.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    // Error bubbles are UI feedback, not something the model ever said -- replaying them
    // as assistant turns would have it apologise for failures it didn't produce.
    const history = messages
      .filter((m) => !m.error)
      .map(({ role, content }) => ({ role, content: String(content).slice(0, MAX_CONTENT) }));
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
            {messages.length > 0 && (
              <button className="icon-btn" onClick={clearMemory}
                      title="Clear this conversation" aria-label="Clear conversation">
                <Trash2 size={14} />
              </button>
            )}
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
                <div className="chat-bubble">{renderChatContent(m.content)}</div>
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
