import { useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";

/**
 * Props:
 * - chats: [{ id, title, msgs }]
 * - onNew: () => void
 * - lang: string
 * - setLang: (v) => void
 * - degree: "bachelor" | "master" | "both"
 * - setDegree: (v) => void
 */
export default function Sidebar({
  chats = [],
  onNew,
  lang,
  setLang,
  degree,
  setDegree,
}) {
  const nav = useNavigate();
  const { id: activeId } = useParams();

  const items = useMemo(
    () =>
      (chats || []).map((c) => ({
        id: c.id,
        title: c.title || "New chat",
      })),
    [chats]
  );

  return (
    <aside className="sidebar">
      {/* top controls */}
      <div className="head">
        <button className="btn primary" style={{ width: "100%" }} onClick={onNew}>
          + New chat
        </button>
      </div>

      {/* preferences */}
      <div style={{ padding: 12, borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "grid", gap: 10 }}>
          <div>
            <label className="muted" style={{ fontSize: 12, display: "block", marginBottom: 6 }}>
              Language
            </label>
            <select
              className="select"
              value={lang}
              onChange={(e) => setLang?.(e.target.value)}
            >
              <option value="en">English</option>
              <option value="he">עברית</option>
              <option value="ar">العربية</option>
            </select>
          </div>

          <div>
            <label className="muted" style={{ fontSize: 12, display: "block", marginBottom: 6 }}>
              Degree
            </label>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
              {["bachelor", "master", "both"].map((d) => (
                <button
                  key={d}
                  onClick={() => setDegree?.(d)}
                  className="btn"
                  style={{
                    padding: "8px 10px",
                    borderRadius: 10,
                    background:
                      degree === d
                        ? "linear-gradient(180deg,#1a2438,#0d1729)"
                        : "linear-gradient(180deg,#0f1729,#0f1729)",
                    borderColor: "var(--border)",
                    textTransform: "capitalize",
                  }}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* recent chats */}
      <div className="scroller">
        <div className="muted" style={{ fontSize: 12, padding: "6px 6px 8px" }}>
          Recent
        </div>
        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 6 }}>
          {items.length === 0 && (
            <li className="muted" style={{ padding: "8px 10px" }}>
              No conversations yet.
            </li>
          )}
          {items.map((c) => {
            const active = c.id === activeId;
            return (
              <li key={c.id}>
                <button
                  className="navbtn"
                  onClick={() => nav(`/chat/${c.id}`)}
                  title={c.title}
                  style={{
                    borderColor: active ? "var(--border)" : "transparent",
                    background: active ? "#0b1224" : "transparent",
                  }}
                >
                  {c.title}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}
