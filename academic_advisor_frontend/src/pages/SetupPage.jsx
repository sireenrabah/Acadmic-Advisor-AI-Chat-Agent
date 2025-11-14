import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/theme.css";
import "../styles/setup.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default function SetupPage() {
  const nav = useNavigate();

  // Fresh defaults (no preload so every session starts clean)
  const [name, setName] = useState("");
  const [lang, setLang] = useState("en");
  const [degree, setDegree] = useState("bachelor");
  const [college, setCollege] = useState("");

  // Optional uploads
  const [bagrut, setBagrut] = useState(null);
  const [psycho, setPsycho] = useState(null);

  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");

  // On mount: hard-reset any previous session/client-side state
  useEffect(() => {
    try {
      Object.keys(localStorage).forEach((k) => {
        if (k.startsWith("aa_")) localStorage.removeItem(k);
      });
    } catch {}
  }, []);

  const isValid = useMemo(() => {
    return Boolean(name.trim() && lang && degree && college);
  }, [name, lang, degree, college]);

  async function uploadDocuments(sessionId) {
    if (!bagrut && !psycho) return null;

    const form = new FormData();
    form.append("session_id", sessionId);
    if (bagrut) form.append("bagrut", bagrut);
    if (psycho) form.append("psychometry", psycho);

    setUploading(true);
    setProgress(30);
    setProgressText("Uploading documents...");
    
    try {
      const r = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      });
      
      setProgress(60);
      setProgressText("Processing documents...");
      
      if (!r.ok) {
        throw new Error(`Upload failed (${r.status})`);
      }
      // ⬇️ Read the server's Bagrut-anchored first question if present
      const data = await r.json().catch(() => ({}));
      
      setProgress(80);
      setProgressText("Documents processed!");
      
      return data?.message || null;
    } finally {
      setUploading(false);
    }
  }

  async function start() {
    if (!isValid || loading) return;
    setError("");
    setLoading(true);
    setProgress(10);
    setProgressText("Initializing session...");
    
    try {
      // Simulate gradual progress
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev < 30) return prev + 2;
          return prev;
        });
      }, 100);
      
      const r = await fetch(`${API_BASE}/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-UI-Lang": lang,
          "Accept-Language": lang,
        },
        body: JSON.stringify({
          language: lang,
          degree,
          name: name.trim(),
          college,
        }),
      });
      
      clearInterval(progressInterval);
      setProgress(40);
      setProgressText("Session created!");
      
      const data = await r.json();

      const sessionId = data.session_id || data.id;
      if (!sessionId) {
        throw new Error("No session_id returned from /start");
      }

      setProgress(50);
      setProgressText("Uploading documents...");

      // Try to upload docs right now; prefer the message produced after Bagrut upload
      const bagrutMsg = await uploadDocuments(sessionId);

      setProgress(80);
      setProgressText("Processing your data...");
      
      // Brief pause to show progress
      await new Promise(resolve => setTimeout(resolve, 500));

      setProgress(95);
      setProgressText("Preparing your interview...");

      const greet = {
        role: "assistant",
        text: bagrutMsg || data.message || "Welcome!",
        ts: Date.now(),
        id: crypto.randomUUID?.() ?? String(Date.now()),
      };

      localStorage.setItem("aa_session_id", sessionId);
      localStorage.setItem("aa_session_msgs", JSON.stringify([greet]));
      localStorage.setItem(
        "aa_chats",
        JSON.stringify([{ id: sessionId, title: "New chat", msgs: [greet] }])
      );
      localStorage.setItem("aa_user_name", name.trim());
      localStorage.setItem("aa_lang", lang);
      localStorage.setItem("aa_degree", degree);
      localStorage.setItem("aa_college", college);

      setProgress(100);
      setProgressText("Ready!");
      
      // Small delay to show 100% before navigating
      setTimeout(() => {
        nav(`/chat/${sessionId}`);
      }, 300);
    } catch (e) {
      setError(
        e?.message ||
          "Could not start a new interview. Please check the backend (/health) and CORS."
      );
      setProgress(0);
      setProgressText("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="setup-wrap">
      <div className="setup">
        <div className="panel card">
          <h1 className="title title-accent">Academic Advisor Interview</h1>
          <p className="muted" style={{ marginTop: 4 }}>
            Upload your documents (optional), enter your details, choose language, degree & college, then start.
          </p>

          {error && (
            <div
              className="card"
              style={{
                marginTop: 10,
                padding: 10,
                borderColor: "rgba(239,68,68,.35)",
                color: "#b91c1c",
                background: "#fff5f5",
              }}
            >
              {error}
            </div>
          )}

          {(loading || uploading) && progress > 0 && (
            <div
              className="card"
              style={{
                marginTop: 12,
                padding: 16,
                borderColor: "rgba(22, 101, 52, 0.25)",
                background: "linear-gradient(to bottom, #ffffff, #fafffe)",
              }}
            >
              <div style={{ 
                fontSize: '14px', 
                fontWeight: 600, 
                color: '#166534',
                marginBottom: 10,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <span>{progressText}</span>
                <span style={{ fontSize: '13px', color: '#6b7280' }}>{progress}%</span>
              </div>
              <div style={{
                width: '100%',
                height: '8px',
                background: 'rgba(22, 101, 52, 0.1)',
                borderRadius: '10px',
                overflow: 'hidden',
                position: 'relative'
              }}>
                <div style={{
                  width: `${progress}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #16a34a, #22c55e)',
                  borderRadius: '10px',
                  transition: 'width 0.5s ease',
                  boxShadow: '0 0 10px rgba(34, 197, 94, 0.4)'
                }} />
              </div>
            </div>
          )}

          <div className="row">
            <div>
              <label className="block text-sm req font-medium text-neutral-700 mb-2">Your name</label>
              <input
                type="text"
                placeholder="e.g., Steve Jobs"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="control"
                aria-required="true"
              />
              <div className="text-xs muted" style={{ marginTop: 6 }}>
                We’ll use your initials for your chat avatar.
              </div>
            </div>

            <div>
              <label className="block text-sm req font-medium text-neutral-700 mb-2">
                College / University
              </label>
              <select
                className="control"
                value={college}
                onChange={(e) => setCollege(e.target.value)}
                aria-required="true"
              >
                <option value="">Select…</option>
                <option value="Ramat Gan Academic College">Ramat Gan Academic College</option>
                <option value="No preference">No preference</option>
              </select>
              <div className="text-xs muted" style={{ marginTop: 6 }}>
                Choose a specific institution or pick <em>No preference</em>.
              </div>
            </div>
          </div>

          <div className="row">
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-2">
                Bagrut (PDF / image)
              </label>
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg"
                onChange={(e) => setBagrut(e.target.files?.[0] || null)}
                className="control"
              />
              {bagrut && (
                <div className="text-xs muted" style={{ marginTop: 6 }}>
                  Selected: {bagrut.name}
                </div>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-2">
                Psychometry (optional)
              </label>
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg"
                onChange={(e) => setPsycho(e.target.files?.[0] || null)}
                className="control"
              />
              {psycho && (
                <div className="text-xs muted" style={{ marginTop: 6 }}>
                  Selected: {psycho.name}
                </div>
              )}
            </div>
          </div>

          <div className="row">
            <div>
              <label className="block text-sm req font-medium text-neutral-700 mb-2">Language</label>
              <select
                className="control"
                value={lang}
                onChange={(e) => setLang(e.target.value)}
                aria-required="true"
              >
                <option value="en">English</option>
                <option value="he">Hebrew</option>
                <option value="ar">Arabic</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="es">Spanish</option>
                <option value="ru">Russian</option>
              </select>
            </div>

            <div>
              <label className="block text-sm req font-medium text-neutral-700 mb-2">Degree</label>
              <div className="pillset" role="group" aria-label="Degree selection">
                {["bachelor", "master", "both"].map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setDegree(d)}
                    className={`pill ${degree === d ? "active" : ""}`}
                    aria-pressed={degree === d}
                  >
                    {d === "both" ? "Both / Not sure" : d[0].toUpperCase() + d.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="footer">
            <button
              className="btn cta"
              onClick={start}
              disabled={loading || uploading || !isValid}
              title={!isValid ? "Please complete all required fields" : undefined}
            >
              {uploading ? "Uploading…" : loading ? "Starting…" : "Start"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
