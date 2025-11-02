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
    try {
      const r = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      });
      if (!r.ok) {
        throw new Error(`Upload failed (${r.status})`);
      }
      // ⬇️ Read the server's Bagrut-anchored first question if present
      const data = await r.json().catch(() => ({}));
      return data?.message || null;
    } finally {
      setUploading(false);
    }
  }

  async function start() {
    if (!isValid || loading) return;
    setError("");
    setLoading(true);
    try {
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
      const data = await r.json();

      const sessionId = data.session_id || data.id;
      if (!sessionId) {
        throw new Error("No session_id returned from /start");
      }

      // Try to upload docs right now; prefer the message produced after Bagrut upload
      const bagrutMsg = await uploadDocuments(sessionId);

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

      nav(`/chat/${sessionId}`);
    } catch (e) {
      setError(
        e?.message ||
          "Could not start a new interview. Please check the backend (/health) and CORS."
      );
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
