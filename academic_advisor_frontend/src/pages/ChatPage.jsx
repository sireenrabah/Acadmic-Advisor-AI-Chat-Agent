import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import useLocal from "../hooks/useLocal";
import "../styles/theme.css";
import "../styles/chat.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const ADVISOR_AVATAR = import.meta.env.VITE_ADVISOR_AVATAR || "/advisor.png";

const uid = () =>
  (crypto?.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));

/* --- helpers --- */
function looksLikeRecommendations(txt = "") {
  const t = String(txt || "").toLowerCase();
  if (!t) return false;
  if (t.includes("recommendation")) return true;
  if (t.includes("final result")) return true;
  if (t.includes("top matches")) return true;
  const numbered = /^\s*\d+[\.\)]\s+/m.test(txt);
  const bulleted = /^\s*[-*•]\s+/m.test(txt);
  const hasPercent = /\b\d{1,3}\s?%/.test(txt);
  const hasScoreWord = /\bscore\b|\bmatch\b/.test(t);
  if ((numbered || bulleted) && (hasPercent || hasScoreWord)) return true;
  if (txt.length > 800 && (numbered || bulleted)) return true;
  return false;
}
function normalizeRecsFromData(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.recommendations)) return data.recommendations;
  if (Array.isArray(data?.results)) return data.results;
  return null;
}

// server history helpers for /turn/next and /turn/answer
function toServerHistory(msgs = []) {
  // [[role, text], ...] with role in {'user','ai'}
  return msgs.map((m) => [m.role === "assistant" ? "ai" : "user", m.text]);
}
function getAskedQuestions(msgs = []) {
  return msgs
    .filter((m) => m.role === "assistant" && m.text?.startsWith("QUESTION:"))
    .map((m) => m.text);
}
function getLastQuestion(msgs = []) {
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === "assistant" && msgs[i].text?.startsWith("QUESTION:")) {
      return msgs[i].text;
    }
  }
  return null;
}

export default function ChatPage() {
  const nav = useNavigate();
  const { id } = useParams();

  const [lang] = useLocal("aa_lang", "en");
  const [degree] = useLocal("aa_degree", "bachelor");
  const [sessionId, setSessionId] = useLocal("aa_session_id", "");
  const [messages, setMessages] = useLocal("aa_session_msgs", []);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [finished, setFinished] = useState(false);

  const listRef = useRef(null);

  // Adopt a fresh session when URL changes (clean slate)
  useEffect(() => {
    if (!id) return;
    if (id !== sessionId) {
      setSessionId(id);
      const fresh = JSON.parse(localStorage.getItem("aa_session_msgs") || "[]");
      setMessages(Array.isArray(fresh) ? fresh : []);
      setFinished(false);
      setError("");
      setInput("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // Boot if user lands directly on /chat without setup
  useEffect(() => {
    let cancelled = false;
    const boot = async () => {
      if (sessionId) return;
      try {
        const r = await fetch(`${API_BASE}/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ language: lang, degree }),
        });
        const data = await r.json();
        if (cancelled) return;
        const greet = {
          role: "assistant",
          text: data.message,
          ts: Date.now(),
          id: uid(),
        };
        const sid = data.session_id || data.id || uid();
        setSessionId(sid);
        setMessages([greet]);
        localStorage.setItem("aa_session_msgs", JSON.stringify([greet]));
      } catch {
        if (!cancelled) setError("Cannot reach backend. Check VITE_API_BASE and /health.");
      }
    };
    boot();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, lang, degree]);

  // Auto-scroll
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length, finished]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading || !sessionId || finished) return;

    setInput("");
    const userMsg = { role: "user", text, ts: Date.now(), id: uid() };
    const nextMsgs = [...messages, userMsg];
    setMessages(nextMsgs);
    localStorage.setItem("aa_session_msgs", JSON.stringify(nextMsgs));

    setLoading(true);
    setError("");
    try {
      // 1) submit the user's answer to the LAST QUESTION
      const lastQ = getLastQuestion(nextMsgs);
      if (lastQ) {
        await fetch(`${API_BASE}/turn/answer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            user_text: text,
            last_question: lastQ,
          }),
        })
          .then((r) => r.json())
          .catch(() => ({}));
      }

      // 2) ask for the NEXT QUESTION (send history + asked_questions)
      const r2 = await fetch(`${API_BASE}/turn/next`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          history: toServerHistory(nextMsgs),
          asked_questions: getAskedQuestions(nextMsgs),
          hint: "",
        }),
      });
      const data2 = await r2.json();

      // Backend returns { text: "QUESTION: ..." } or { text: "__RECOMMEND__\nTransition\n\nTop 3 majors..." }
      let aiText = data2?.text || "";

      // compatibility if someone returns {message: "..."}
      if (!aiText) {
        aiText = data2?.message || "";
      }

      // Check if backend signals to show recommendations
      if (aiText.startsWith("__RECOMMEND__")) {
        // Remove the signal prefix
        const content = aiText.replace("__RECOMMEND__\n", "").trim();
        
        if (content) {
          // Show the transition message + inline recommendations in chat
          const aiMsg = { role: "assistant", text: content, ts: Date.now(), id: uid() };
          const withRecs = [...nextMsgs, aiMsg];
          setMessages(withRecs);
          localStorage.setItem("aa_session_msgs", JSON.stringify(withRecs));
        }
        
        // Navigate to results after a delay (so user can read inline recs)
        setTimeout(() => nav(`/results/${sessionId}`), 2500);
        setLoading(false);
        return;
      }

      const aiMsg = { role: "assistant", text: aiText || "…", ts: Date.now(), id: uid() };
      const after = [...nextMsgs, aiMsg];
      setMessages(after);
      localStorage.setItem("aa_session_msgs", JSON.stringify(after));
    } catch {
      const aiMsg = {
        role: "assistant",
        text: "Network error. Please try again.",
        ts: Date.now(),
        id: uid(),
      };
      const afterErr = [...nextMsgs, aiMsg];
      setMessages(afterErr);
      localStorage.setItem("aa_session_msgs", JSON.stringify(afterErr));
      setError("Request failed");
    } finally {
      setLoading(false);
    }
  };

  // Exit = JUST exit, no recommendations; clear state and go to Start
  const exitInterview = async () => {
    try {
      await fetch(`${API_BASE}/finish`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => {});
    } finally {
      localStorage.removeItem("aa_session_id");
      localStorage.removeItem("aa_session_msgs");
      localStorage.setItem("aa_chats", "[]");
      setMessages([]);
      setFinished(false);
      nav("/", { replace: true });
    }
  };

  return (
    <div className="app">
      <main className="main">
        <div className="topbar">
          <div className="inner container">
            <div className="brand">
              <div className="logo">AA</div>
              <div style={{ fontWeight: 700 }}>Academic Advisor — Interview</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div className="badge">
                Degree: <strong className="badge-strong">{degree}</strong>
              </div>
              <button className="btn outline" onClick={exitInterview} disabled={!sessionId}>
                Exit
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="container" style={{ marginTop: 12 }}>
            <div
              className="card"
              style={{
                padding: 10,
                borderColor: "rgba(239,68,68,.35)",
                background: "#fff5f5",
                color: "#b91c1c",
              }}
            >
              {error}
            </div>
          </div>
        )}

        <div ref={listRef} className="stream">
          <div className="inner">
            {messages.map((m) => (
              <Message key={m.id} role={m.role} text={m.text} />
            ))}

            {loading && <Typing />}

            {/* Finish Card (optional future use) */}
            {finished && (
              <div className="msgrow">
                <div className="avatar advisor">
                  <img src={ADVISOR_AVATAR} alt="Academic Advisor" />
                </div>
                <div
                  className="bubble"
                  style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}
                >
                  <div>
                    <div style={{ fontWeight: 700 }}>Well done! The interview is complete.</div>
                    <div className="muted" style={{ marginTop: 4 }}>
                      Click below to view your final recommendations.
                    </div>
                  </div>
                  <button className="btn cta" onClick={() => nav(`/results/${sessionId}`)}>
                    See Recommendations
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="composer">
          <div className="inner">
            <div className="inputbar">
              <textarea
                className="textarea"
                placeholder="Message the Academic Advisor…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                disabled={finished}
              />
              <button className="btn cta" onClick={send} disabled={!input.trim() || loading || finished}>
                Send
              </button>
            </div>
            <div className="hint">
              One-on-one interview. Click <strong>Exit</strong> to leave without viewing recommendations.
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

/* --- Avatars & Message components --- */
function getInitials(name) {
  const n = (name || "").trim();
  if (!n) return "you";
  const parts = n.split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toLowerCase();
  const w = parts[0];
  return w.slice(0, 2).padEnd(2, w[0]).toLowerCase();
}

function AdvisorAvatar() {
  const [imgOk, setImgOk] = useState(true);
  return imgOk ? (
    <div className="avatar advisor" aria-label="Academic Advisor">
      <img
        src={ADVISOR_AVATAR}
        alt="Academic Advisor"
        onError={() => setImgOk(false)}
        loading="eager"
        decoding="async"
      />
    </div>
  ) : (
    <div className="avatar advisor fallback" aria-label="Academic Advisor">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 3 1 8l11 5 8-3.636V16h2V8L12 3zM3 13v4l9 4 9-4v-4l-9 4-9-4z" />
      </svg>
    </div>
  );
}

function UserAvatar() {
  const name = localStorage.getItem("aa_user_name") || "";
  const initials = getInitials(name);
  return (
    <div className="avatar user" aria-label={name || "You"}>
      {initials}
    </div>
  );
}

function Message({ role, text }) {
  const isUser = role === "user";
  return (
    <div className={`msgrow ${isUser ? "user" : ""}`}>
      {!isUser && <AdvisorAvatar />}
      <div className={`bubble ${isUser ? "user" : ""}`}>{text}</div>
      {isUser && <UserAvatar />}
    </div>
  );
}

function Typing() {
  return (
    <div className="msgrow">
      <AdvisorAvatar />
      <div className="bubble">
        <span className="typing">
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="dot"></span>
        </span>
      </div>
    </div>
  );
}
