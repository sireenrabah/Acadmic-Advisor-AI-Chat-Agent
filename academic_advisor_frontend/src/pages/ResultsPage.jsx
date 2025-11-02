import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import "../styles/theme.css";
import "../styles/results.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

/* ------------ helpers ------------ */
function normalizeRecommendations(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.recommendations)) return data.recommendations;
  if (Array.isArray(data?.results)) return data.results;
  return [];
}
function toPercent(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return null;
  const val = score <= 1 ? score * 100 : score;
  return Math.max(0, Math.min(100, Math.round(val)));
}

/** Parse raw 3-item text into a structured array */
function parseTextRecommendations(raw = "") {
  if (!raw) return [];
  const lines = raw.split(/\r?\n/);
  const blocks = [];
  let cur = null;

  const startRe = /^\s*(\d+)[\.\)]\s+(.*?)\s*\(match\s*([\d.]+)%\)/i;
  const bulletRe = /^\s*(?:[-*•])\s+(.*)$/;

  for (const ln of lines) {
    const s = ln.trimEnd();
    const start = s.match(startRe);
    if (start) {
      if (cur) blocks.push(cur);
      cur = {
        rank: Number(start[1]),
        name: start[2].trim(),
        score: Number(start[3]),
        bullets: [],
        source: "",
        components: {},
      };
      continue;
    }
    if (!cur) continue;
    const b = s.match(bulletRe);
    if (b) {
      const text = b[1].trim();
      const src = text.match(/^Sources:\s*(.+)$/i);
      if (src) { cur.source = src[1].trim(); continue; }
      const comps = text.match(/Match\s+(\d{1,3})%.*?\bcos\s+(\d{1,3})%.*?\brubric\s+(\d{1,3})%.*?\bBagrut\s+(\d{1,3})%/i);
      if (comps) {
        cur.components = {
          match: Number(comps[1]),
          cos: Number(comps[2]),
          rubric: Number(comps[3]),
          bagrut: Number(comps[4]),
        };
        continue;
      }
      cur.bullets.push(text);
    }
  }
  if (cur) blocks.push(cur);

  return blocks
    .sort((a, b) => (a.rank || 99) - (b.rank || 99))
    .slice(0, 3)
    .map((b) => ({
      name: b.name,
      score: b.score,
      summary: b.bullets[0] || "",
      reasons: b.bullets.slice(1, 4),
      source: b.source || "",
      components: b.components,
    }));
}

export default function ResultsPage() {
  const nav = useNavigate();
  const { id } = useParams(); // session_id

  const [recs, setRecs] = useState([]);
  const [raw, setRaw] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const parsedFromRaw = useMemo(() => parseTextRecommendations(raw), [raw]);
  const finalRecs = recs.length ? recs : parsedFromRaw;

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError("");
      try {
        const cachedArr = localStorage.getItem(`aa_cached_recs_${id}`);
        const cachedText = localStorage.getItem(`aa_cached_recs_text_${id}`);
        const arr = cachedArr ? JSON.parse(cachedArr) : null;
        if (!cancelled && Array.isArray(arr) && arr.length > 0) {
          setRecs(arr); setRaw(""); setLoading(false); return;
        }
        if (!cancelled && cachedText && cachedText.trim()) {
          setRecs([]); setRaw(cachedText); setLoading(false); return;
        }
      } catch {}

      try {
        const r = await fetch(`${API_BASE}/recommendations?session_id=${encodeURIComponent(id)}`);
        const data = await r.json();
        const arr = normalizeRecommendations(data);
        if (!cancelled) { setRecs(arr); setRaw(""); }
        if (Array.isArray(arr) && arr.length > 0) {
          localStorage.setItem(`aa_cached_recs_${id}`, JSON.stringify(arr));
          localStorage.removeItem(`aa_cached_recs_text_${id}`);
        } else if (!cancelled && typeof data?.message === "string" && parseTextRecommendations(data.message).length) {
          localStorage.setItem(`aa_cached_recs_text_${id}`, data.message);
          setRaw(data.message);
        }
      } catch {
        if (!cancelled) setError("Could not load recommendations.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [id]);

  const empty = !loading && !error && finalRecs.length === 0;

  return (
    <div className="results-wrap">
      <div className="results container">
        {/* Global page title (no card) */}
        <div className="results-header">
          <h1 className="page-title accent">Final Recommendations</h1>
          <p className="muted">Based on your interview responses and signals, here are your top matches.</p>
        </div>

        {loading && <div className="card loader">Loading recommendations…</div>}
        {error && <div className="card error">{error}</div>}
        {empty && <div className="card empty">No recommendations available for this session.</div>}

        {finalRecs.length > 0 && (
          <div className="grid">
            {finalRecs.slice(0, 3).map((m, idx) => {
              const name = m?.name || m?.english_name || "Major";
              const pct = toPercent(m?.score) ?? toPercent(m?.match) ?? toPercent(m?.components?.match);
              const reasons = Array.isArray(m?.reasons) && m.reasons.length ? m.reasons : [];
              const summary = m?.summary || "";
              const source = m?.source || "";

              return (
                <div key={idx} className="rec-card card">
                  <div className="rec-head">
                    <span className="badge-rank">#{idx + 1}</span>
                    <h2 className="rec-title">{name}</h2>
                  </div>

                  {pct != null && (
                    <div className="scorerow" aria-label={`Match score ${pct}%`}>
                        <span className="scorelabel">Match</span>
                        <div className="scorebar">
                        <div className="fill" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="scorepct">{pct}%</span>
                    </div>
                    )}


                  {summary && <p className="muted" style={{ marginTop: 8 }}>{summary}</p>}

                  {reasons.length > 0 && (
                    <ul className="reasons">
                      {reasons.slice(0, 4).map((r, i) => <li key={i}>{r}</li>)}
                    </ul>
                  )}

                  <div className="rec-footer">
                    {m?.dept && <span className="chip">{m.dept}</span>}
                    {m?.degree_level && <span className="chip">{m.degree_level}</span>}
                    {source && <span className="chip light">Source: {source}</span>}
                    {Array.isArray(m?.keywords) && m.keywords.slice(0,3).map((k, i) => (
                      <span className="chip light" key={i}>{k}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="foot-actions">
          <button className="btn cta" onClick={() => nav("/")}>Start New Interview</button>
        </div>
      </div>
    </div>
  );
}
