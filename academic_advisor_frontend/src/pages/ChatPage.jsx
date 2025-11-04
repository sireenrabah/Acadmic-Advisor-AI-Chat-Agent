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
  console.log("[getLastQuestion] Called with", msgs.length, "messages");
  
  // Search backwards for the last assistant message
  // Check ORIGINAL text (before stripping) for "QUESTION:" prefix
  for (let i = msgs.length - 1; i >= 0; i--) {
    const msg = msgs[i];
    console.log(`[getLastQuestion] Checking msg[${i}]: role=${msg.role}, text_start="${msg.text?.substring(0, 30)}..."`);
    
    if (msg.role === "assistant") {
      // Check if text starts with "QUESTION:" OR if it's a regular question
      // (some questions might not have prefix after stripping for display)
      const originalText = msg.text || "";
      
      if (originalText.startsWith("QUESTION:")) {
        console.log("[getLastQuestion] Found QUESTION: prefix at index", i);
        return originalText;  // Return WITH prefix for backend
      }
      
      // Fallback: if it's an assistant message and doesn't start with __RECOMMEND__, treat as question
      if (!originalText.startsWith("__RECOMMEND__")) {
        console.log("[getLastQuestion] Found assistant message (no QUESTION: prefix) at index", i, "- re-adding prefix");
        // Re-add prefix if missing (defensive programming)
        return `QUESTION: ${originalText}`;
      }
      
      console.log("[getLastQuestion] Skipping __RECOMMEND__ message at index", i);
    }
  }
  
  console.error("[getLastQuestion] NO question found in", msgs.length, "messages!");
  return null;
}

// Helper to strip "QUESTION: " prefix for display only
function stripQuestionPrefix(text) {
  if (text?.startsWith("QUESTION: ")) {
    return text.slice("QUESTION: ".length);
  }
  return text;
}

// Helper to parse recommendations from __RECOMMEND__ text
function parseRecommendationsFromText(text) {
  /*
  Expected format (Hebrew):
  1. תואר ראשון B.Sc. :במדעי המחשב (90.1% התאמה)
     Rationale text.
  
  OR (English):
  1. Computer Science B.Sc. (90.1% Match)
     Rationale text.
  */
  const lines = text.split('\n');
  const majors = [];
  let currentMajor = null;
  
  // Updated regex: captures "1. Major Name (90.1%" and stops before "התאמה" or "Match"
  // Matches: "1. " + major name + " (" + number + "%" (optional space) + rest
  const majorRegex = /^(\d+)\.\s+(.+?)\s+\((\d+(?:\.\d+)?)\s*%/;
  
  for (const line of lines) {
    const match = line.match(majorRegex);
    if (match) {
      // Save previous major if exists
      if (currentMajor) {
        majors.push(currentMajor);
      }
      
      // Start new major
      // Remove trailing Hebrew/English match words from major name
      let majorName = match[2].trim();
      // Strip any trailing "התאמה)" or "Match)" that might have leaked into capture group 2
      majorName = majorName.replace(/\s*(התאמה|Match)\)?$/i, '');
      
      currentMajor = {
        rank: parseInt(match[1]),
        original_name: majorName,
        english_name: majorName, // Same as original for now
        score: parseFloat(match[3])
      };
    }
  }
  
  // Add last major
  if (currentMajor) {
    majors.push(currentMajor);
  }
  
  console.log("[parseRecommendationsFromText] Parsed", majors.length, "majors:", majors);
  return majors;
}

// Helper to detect if user is requesting major details
function isMajorInfoRequest(text, hasMajors) {
  if (!hasMajors || !text) return false;
  
  const t = text.trim().toLowerCase();
  
  // Check if it's just a number (1, 2, or 3)
  if (/^[123]$/.test(t)) return true;
  
  // Check if it contains "yes" or confirmation words
  if (t === 'yes' || t === 'כן' || t === 'نعم') return true;
  
  // Check if text contains multiple words (might be a major name)
  if (t.split(/\s+/).length >= 2) return true;
  
  return false;
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
  const [recommendedMajors, setRecommendedMajors] = useState([]); // Store majors after recommendations

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
      // CHECK: If user is requesting major details after recommendations
      if (isMajorInfoRequest(text, recommendedMajors.length > 0)) {
        console.log("[ChatPage] Detected major info request:", text);
        console.log("[ChatPage] Available majors:", recommendedMajors);
        
        const r = await fetch(`${API_BASE}/turn/major_info`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            user_text: text,
            recommended_majors: recommendedMajors
          }),
        });
        
        const data = await r.json();
        const aiText = data?.text || "Sorry, I couldn't process that request.";
        
        const aiMsg = { role: "assistant", text: aiText, ts: Date.now(), id: uid() };
        const after = [...nextMsgs, aiMsg];
        setMessages(after);
        localStorage.setItem("aa_session_msgs", JSON.stringify(after));
        setLoading(false);
        return;
      }
      
      // NORMAL FLOW: Regular interview question/answer
      // 1) submit the user's answer to the LAST QUESTION
      const lastQ = getLastQuestion(nextMsgs);
      console.log("[ChatPage] getLastQuestion returned:", lastQ ? lastQ.substring(0, 80) : "NULL");
      console.log("[ChatPage] nextMsgs length:", nextMsgs.length);
      console.log("[ChatPage] Last 3 messages:", nextMsgs.slice(-3).map(m => ({ role: m.role, text: m.text?.substring(0, 40) })));
      
      console.log("[ChatPage:CRITICAL] About to check lastQ. Value:", lastQ ? `"${lastQ.substring(0, 80)}..."` : "NULL/UNDEFINED");
      
      if (lastQ) {
        const payload = {
          session_id: sessionId,
          user_text: text,
          last_question: lastQ,
        };
        console.log("[ChatPage:CALLING] /turn/answer with payload:", JSON.stringify(payload).substring(0, 200));
        
        try {
          const response = await fetch(`${API_BASE}/turn/answer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          
          console.log("[ChatPage:RESPONSE] /turn/answer status:", response.status, response.statusText);
          
          const data = await response.json();
          console.log("[ChatPage:DATA] /turn/answer returned:", data);
        } catch (err) {
          console.error("[ChatPage:ERROR] /turn/answer failed:", err.message, err.stack);
        }
      } else {
        console.error("[ChatPage:BUG] No last question found - /turn/answer NOT called!");
        console.error("[ChatPage:BUG] nextMsgs dump:", JSON.stringify(nextMsgs.map(m => ({ role: m.role, text_length: m.text?.length, starts_with: m.text?.substring(0, 20) }))));
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
        
        // Parse majors from the text for later major info requests
        const parsedMajors = parseRecommendationsFromText(content);
        console.log("[ChatPage] Parsed recommendations:", parsedMajors);
        setRecommendedMajors(parsedMajors);
        
        if (content) {
          // Show the transition message + inline recommendations in chat
          const aiMsg = { role: "assistant", text: content, ts: Date.now(), id: uid() };
          const withRecs = [...nextMsgs, aiMsg];
          setMessages(withRecs);
          localStorage.setItem("aa_session_msgs", JSON.stringify(withRecs));
        }
        
        // DON'T auto-navigate! Let user explore major details first
        // User can manually navigate or type major numbers
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
  // Strip "QUESTION: " prefix from advisor messages for display
  const displayText = !isUser ? stripQuestionPrefix(text) : text;
  return (
    <div className={`msgrow ${isUser ? "user" : ""}`}>
      {!isUser && <AdvisorAvatar />}
      <div className={`bubble ${isUser ? "user" : ""}`}>{displayText}</div>
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
