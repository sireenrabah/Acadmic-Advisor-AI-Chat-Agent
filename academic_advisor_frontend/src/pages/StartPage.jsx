import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import bot from "../assets/bot-hello.png";
import "../styles/theme.css";
import "../styles/start.css";

export default function StartPage() {
  const nav = useNavigate();

  return (
    <div className="start-page">
      {/* If your Header is fixed globally and theme.css already provides .page-offset,
          you can keep this wrapper. Otherwise you can remove it safely. */}
      <div className="page-offset">
        <div className="container">
          {/* HERO */}
          <section className="card hero hero-card">
            <div>
              <div className="pill">New • Smarter major matching</div>
              <h1 style={{ margin: "12px 0 6px", fontSize: 34, lineHeight: 1.15 }}>
                Your personalized academic advisor
              </h1>
              <p className="muted" style={{ margin: 0 }}>
                A short, adaptive interview that learns how you think—then suggests
                3 majors that fit you best. You can later attach your
                <strong> Bagrut</strong> & <strong> Psychometric</strong> PDFs on the setup page
                to improve accuracy.
              </p>

              <div style={{ display: "flex", gap: 10, marginTop: 18, flexWrap: "wrap" }}>
                <div className="chip">⚡ 8–12 quick questions</div>
                <div className="chip">🎯 Tailored recommendations</div>
                <div className="chip">🔐 Private</div>
              </div>

              <div style={{ marginTop: 22 }}>
                <button className="btn cta" onClick={() => nav("/setup")}>
                  Start
                </button>
              </div>
            </div>

            {/* IMAGE */}
            <div className="hero-art">
              <img
                src={bot}
                alt="Friendly chatbot waving next to a chat window"
                style={{
                  width: "100%",
                  height: "auto",
                  display: "block",
                  filter: "drop-shadow(0 12px 28px rgba(0,0,0,.25))",
                  borderRadius: 16,
                }}
              />
            </div>
          </section>

          {/* FEATURES */}
          <section className="grid3" style={{ marginTop: 18 }}>
            <div className="card feature">
              <div className="feature-emoji">🧭</div>
              <h3>Guided, not graded</h3>
              <p className="muted">Scenario prompts about planning, logic & patterns—no trick quizzes.</p>
            </div>
            <div className="card feature">
              <div className="feature-emoji">🧠</div>
              <h3>Thinking profile</h3>
              <p className="muted">We map your answers to majors’ cognitive demands to find fit.</p>
            </div>
            <div className="card feature">
              <div className="feature-emoji">📎</div>
              <h3>Better with docs</h3>
              <p className="muted">On the setup page, attach Bagrut & Psychometric to boost accuracy.</p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
