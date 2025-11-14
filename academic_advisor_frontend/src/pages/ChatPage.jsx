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
  }, [messages.length, finished, loading]);

  // Handle follow-up option button clicks
  const handleOptionClick = async (option) => {
    if (loading || !sessionId || finished) return;
    
    // Map option to user-friendly text (for display only)
    const optionTexts = {
      '1': lang === 'he' ? '1️⃣ למה זה מתאים לי?' : '1️⃣ Why is this good for me?',
      '2': lang === 'he' ? '2️⃣ דרישות קבלה וחוסרים' : '2️⃣ Eligibility & Requirements',
      '3': lang === 'he' ? '3️⃣ קורסים לדוגמה' : '3️⃣ Sample Courses',
      '4': lang === 'he' ? '4️⃣ מסלולי קריירה' : '4️⃣ Career Paths'
    };
    
    const text = optionTexts[option] || option;
    const userMsg = { role: "user", text, ts: Date.now(), id: uid() };
    const nextMsgs = [...messages, userMsg];
    setMessages(nextMsgs);
    localStorage.setItem("aa_session_msgs", JSON.stringify(nextMsgs));

    setLoading(true);
    setError("");
    
    try {
      // Send option selection to backend
      const r = await fetch(`${API_BASE}/turn/followup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          option: option,  // Send just the number/option code
          majors: recommendedMajors
        }),
      });
      
      if (!r.ok) throw new Error(`followup failed: ${r.status}`);
      const data = await r.json();
      
      const botMsg = {
        role: "assistant",
        text: data.text || data.message || "I'll provide more details soon.",
        ts: Date.now(),
        id: uid(),
      };
      
      const finalMsgs = [...nextMsgs, botMsg];
      setMessages(finalMsgs);
      localStorage.setItem("aa_session_msgs", JSON.stringify(finalMsgs));
    } catch (err) {
      setError(err?.message || "Could not process your selection.");
    } finally {
      setLoading(false);
    }
  };

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
      // CHECK 1: If user typed ANY number after recommendations, use followup endpoint
      // Backend will validate if the number is in valid range
      if (/^\d+$/.test(text) && recommendedMajors.length > 0) {
        console.log("[ChatPage] Detected followup option:", text);
        
        const r = await fetch(`${API_BASE}/turn/followup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            option: text,
            majors: recommendedMajors
          }),
        });
        
        if (!r.ok) throw new Error(`followup failed: ${r.status}`);
        const data = await r.json();
        const aiText = data?.text || "Sorry, I couldn't process that request.";
        
        const aiMsg = { role: "assistant", text: aiText, ts: Date.now(), id: uid() };
        const after = [...nextMsgs, aiMsg];
        setMessages(after);
        localStorage.setItem("aa_session_msgs", JSON.stringify(after));
        setLoading(false);
        return;
      }
      
      // CHECK 2: If user is requesting major details after recommendations
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
          
          // Add follow-up question with options
          const followUpText = lang === 'he' 
            ? 'איזו מגמה מעניינת אותך? הקלד את המספר (1, 2, או 3)'
            : 'Which major interests you? Type the number (1, 2, or 3)';
          const followUpMsg = { role: "assistant", text: followUpText, ts: Date.now() + 1, id: uid() };
          
          const withRecs = [...nextMsgs, aiMsg, followUpMsg];
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
              <Message key={m.id} role={m.role} text={m.text} onOptionClick={handleOptionClick} />
            ))}

            {/* Typing Indicator */}
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
                dir="auto"
                style={{ textAlign: /[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/.test(input) ? "right" : "left" }}
              />
              <button className="btn cta" onClick={send} disabled={!input.trim() || loading || finished}>
                Send
              </button>
            </div>

            {/* End Conversation Button - Shows after recommendations */}
            {recommendedMajors.length > 0 && (
              <div style={{ marginTop: 12, textAlign: "center" }}>
                <button 
                  className="btn cta"
                  onClick={() => {
                    // Store summary data in localStorage for ResultsPage
                    localStorage.setItem("aa_final_summary", JSON.stringify({
                      sessionId,
                      recommendedMajors,
                      messages,
                      language: lang,
                      degree
                    }));
                    nav(`/results/${sessionId}`);
                  }}
                  style={{
                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    padding: "12px 32px",
                    fontSize: "16px",
                    fontWeight: 600,
                    width: "100%"
                  }}
                >
                  {lang === 'he' ? '📄 סיים שיחה וקבל סיכום' : lang === 'ar' ? '📄 إنهاء المحادثة والحصول على ملخص' : '📄 End Conversation & Get Summary'}
                </button>
              </div>
            )}

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

function Message({ role, text, onOptionClick }) {
  const isUser = role === "user";
  // Strip "QUESTION: " prefix from advisor messages for display
  const displayText = !isUser ? stripQuestionPrefix(text) : text;
  
  // Detect if text contains Hebrew or Arabic characters
  const isRTL = /[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/.test(displayText);
  
  // Check if this is a recommendations message (contains numbered list with percentages)
  // Look for pattern ANYWHERE in text, not just at start
  const isRecommendations = !isUser && /\n\d+\.\s+.+?\s+\(\d+(?:\.\d+)?\s*%/.test(displayText);
  
  // Check if this is a follow-up question (appears after recommendations)
  const isFollowUpQuestion = !isUser && !isRecommendations && (
    /איזו מגמה מעניינת|הקלד את המספר/.test(displayText) ||
    /which major interests|type the number/i.test(displayText) ||
    /בחרת:.*מה תרצה לדעת/.test(displayText) ||
    /you selected:.*what would you like/i.test(displayText)
  );
  
  // Format the message content with proper styling
  const formatMessage = (msg) => {
    // Strip markdown bold formatting (** or __)
    const cleanMsg = msg.replace(/\*\*(.*?)\*\*/g, '$1').replace(/__(.*?)__/g, '$1');
    
    const lines = cleanMsg.split('\n');
    
    if (!isRecommendations) {
      // Regular message - detect multi-part greeting (college greeting + bagrut + question)
      // College greeting pattern: short (< 150 chars), not a question, likely first part of 3+ part message
      const parts = cleanMsg.split('\n\n');
      
      if (parts.length >= 3) {
        const firstPart = parts[0].trim();
        // Check if first part looks like a college/welcome greeting (short, no question mark)
        if (firstPart.length < 150 && !firstPart.endsWith('?')) {
          // Render first part BOLD as separate visual section
          return (
            <>
              <div style={{ 
                fontWeight: 700, 
                fontSize: '16px',
                marginBottom: '12px',
                color: '#166534',
                lineHeight: '1.5'
              }}>
                {firstPart}
              </div>
              <div style={{ fontSize: '14.5px', lineHeight: '1.6', color: '#1f2937' }}>
                {parts.slice(1).join('\n\n')}
              </div>
            </>
          );
        }
      }
      
      // Check if this is a follow-up question - render BOLD (no buttons)
      if (isFollowUpQuestion) {
        return (
          <div style={{ 
            fontWeight: 700, 
            fontSize: '15px',
            color: '#1f2937',
            lineHeight: '1.6',
            whiteSpace: 'pre-line'
          }}>
            {cleanMsg}
          </div>
        );
      }
      
      return cleanMsg;
    }
    
    // Format recommendations with beautiful styling
    const formatted = [];
    let introText = [];
    let inMajorsList = false;
    let currentMajor = null;
    let currentRationale = [];
    
    lines.forEach((line, idx) => {
      const trimmed = line.trim();
      
      // Match: "1. Major Name (95.5% Match)" or "1. Major Name (95.5% התאמה)"
      const majorMatch = trimmed.match(/^(\d+)\.\s+(.+?)\s+\((\d+(?:\.\d+)?)\s*%\s*(.+?)\)$/);
      
      if (majorMatch) {
        // Save previous major if exists
        if (currentMajor) {
          formatted.push(
            <div key={`major-${formatted.length}`} style={{ 
              marginBottom: '20px', 
              paddingBottom: '16px', 
              borderBottom: '1px solid rgba(22, 101, 52, 0.15)'
            }}>
              {currentMajor}
              {currentRationale.length > 0 && (
                <div style={{ 
                  fontSize: '14px', 
                  color: '#6b7280',
                  lineHeight: '1.6',
                  marginTop: '8px',
                  paddingLeft: '0px',
                  paddingRight: isRTL ? '0px' : '8px'
                }}>
                  {currentRationale.join(' ')}
                </div>
              )}
            </div>
          );
          currentRationale = [];
        }
        
        // Add intro text if we haven't entered majors list yet
        if (!inMajorsList && introText.length > 0) {
          formatted.push(
            <div key="intro" style={{ 
              fontWeight: 600, 
              marginBottom: '16px', 
              fontSize: '15px',
              color: '#1f2937',
              lineHeight: '1.5'
            }}>
              {introText.join('\n')}
            </div>
          );
          introText = [];
          inMajorsList = true;
        }
        
        const rank = majorMatch[1];
        const majorName = majorMatch[2].trim();
        const score = parseFloat(majorMatch[3]);
        const matchText = majorMatch[4]; // "Match" or "התאמה"
        
        // Create progress bar color based on score
        const getScoreColor = (score) => {
          if (score >= 90) return { bg: '#166534', text: '#ffffff' };
          if (score >= 80) return { bg: '#16a34a', text: '#ffffff' };
          if (score >= 70) return { bg: '#22c55e', text: '#ffffff' };
          return { bg: '#86efac', text: '#166534' };
        };
        
        const colors = getScoreColor(score);
        
        currentMajor = (
          <>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', marginBottom: '10px' }}>
              <span style={{ 
                fontSize: '20px', 
                fontWeight: 700, 
                color: '#166534',
                minWidth: '28px',
                lineHeight: '1.3'
              }}>
                {rank}.
              </span>
              <div style={{ flex: 1 }}>
                <div style={{ 
                  fontSize: '17px', 
                  fontWeight: 700, 
                  color: '#1f2937',
                  lineHeight: '1.4',
                  marginBottom: '10px'
                }}>
                  {majorName}
                </div>
                
                {/* Progress bar */}
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '10px',
                  marginTop: '8px'
                }}>
                  <div style={{ 
                    flex: 1,
                    height: '8px',
                    background: '#f3f4f6',
                    borderRadius: '10px',
                    overflow: 'hidden',
                    position: 'relative'
                  }}>
                    <div style={{ 
                      width: `${score}%`,
                      height: '100%',
                      background: `linear-gradient(90deg, ${colors.bg}, ${colors.bg}dd)`,
                      borderRadius: '10px',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  <span style={{ 
                    fontSize: '16px', 
                    fontWeight: 700,
                    color: colors.bg,
                    minWidth: '70px',
                    textAlign: isRTL ? 'left' : 'right'
                  }}>
                    {score}% {matchText}
                  </span>
                </div>
              </div>
            </div>
          </>
        );
      } else if (trimmed && inMajorsList) {
        // This is a rationale line - accumulate it
        currentRationale.push(trimmed);
      } else if (trimmed && !inMajorsList) {
        // This is intro text before the list
        introText.push(trimmed);
      }
    });
    
    // Add last major
    if (currentMajor) {
      formatted.push(
        <div key={`major-${formatted.length}`} style={{ 
          marginBottom: '20px', 
          paddingBottom: '16px', 
          borderBottom: 'none'
        }}>
          {currentMajor}
          {currentRationale.length > 0 && (
            <div style={{ 
              fontSize: '14px', 
              color: '#6b7280',
              lineHeight: '1.6',
              marginTop: '8px',
              paddingLeft: '0px',
              paddingRight: isRTL ? '0px' : '8px'
            }}>
              {currentRationale.join(' ')}
            </div>
          )}
        </div>
      );
    }
    
    return <div>{formatted}</div>;
  };
  
  return (
    <div className={`msgrow ${isUser ? "user" : ""}`}>
      {!isUser && <AdvisorAvatar />}
      <div 
        className={`bubble ${isUser ? "user" : ""} ${isRecommendations ? "recommendations" : ""}`}
        dir={isRTL ? "rtl" : "ltr"}
        style={isRTL ? { textAlign: "right" } : {}}
      >
        {formatMessage(displayText)}
      </div>
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
