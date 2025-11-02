# generation/generator.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import json

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# Use your rule checker so the LLM can ground on a computed pass/fail signal
from query.bagrut_features import eligibility_flags_from_rules


def _detect_intent(q: str) -> str:
    """
    Very light intent routing to help the prompt:
      - 'eligibility'  -> asks if the student qualifies / requirements / conditions
      - 'courses'      -> asks about curriculum / study plan / classes / structure
      - 'roles'        -> asks about jobs / careers / what you can do after
      - 'general'      -> everything else
    """
    t = (q or "").lower()
    if any(k in t for k in ["eligible", "eligibility", "requirements", "admission", "acceptance", "qualify", "conditions"]):
        return "eligibility"
    if any(k in t for k in ["course", "curriculum", "classes", "modules", "study plan", "syllabus", "semester"]):
        return "courses"
    if any(k in t for k in ["job", "role", "career", "positions", "industry", "work", "after graduating", "future"]):
        return "roles"
    return "general"


class Generator:
    """
    Majors Q&A generator (eligibility / courses / future roles).
    Language is controlled ONLY by the `ui_language` arg you pass in.
    No hard-coded branching by language.
    """

    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None, vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore  # optional: if you later want to retrieve PDF snippets

    # -------- context packing -------------------------------------------------
    def _build_context(self, major: Dict[str, Any]) -> str:
        parts: List[str] = []
        en = (major.get("english_name") or "").strip()
        orig = (major.get("original_name") or "").strip()
        if en or orig:
            parts.append(f"Major: {en or orig}")
        if major.get("keywords"):
            parts.append("Keywords: " + ", ".join(major.get("keywords")))
        if major.get("sample_courses"):
            parts.append("Sample courses: " + ", ".join(major.get("sample_courses")))
        # Put raw rules for transparency; LLM should summarize, not invent.
        if major.get("eligibility_rules"):
            try:
                pretty = json.dumps(major.get("eligibility_rules"), ensure_ascii=False)
            except Exception:
                pretty = str(major.get("eligibility_rules"))
            parts.append("Eligibility rules (raw): " + pretty)
        return "\n".join(parts).strip()

    # -------- public: answer --------------------------------------------------
    def answer(
        self,
        *,
        ui_language: str,
        major: Dict[str, Any],
        bagrut_json: Dict[str, Any],
        question: str
    ) -> str:
        # Safe fallback if LLM not wired
        if not self.llm:
            intent = _detect_intent(question)
            if intent == "eligibility":
                ok, msgs = eligibility_flags_from_rules(major.get("eligibility_rules") or {}, bagrut_json or {})
                status = "PASS" if ok else "MAY NOT MEET ALL REQUIREMENTS"
                notes = ("; ".join(msgs)) if msgs else "No detailed rules available."
                return f"[Eligibility: {status}] {notes}"
            if intent == "courses":
                return "Typical structure: core + electives + possibly labs/projects. Connect with the department for exact plan."
            if intent == "roles":
                return "Common roles depend on the major’s focus; typical paths include industry, research, or interdisciplinary roles."
            return "Ask about eligibility, courses, or future roles for this major."

        context = self._build_context(major)
        intent = _detect_intent(question)

        # Compute rule-based eligibility once and pass to the prompt as a clear signal.
        ok, msgs = eligibility_flags_from_rules(major.get("eligibility_rules") or {}, bagrut_json or {})
        elig_status = "PASS" if ok else "GAPS_OR_UNVERIFIED"
        elig_notes = "; ".join(msgs) if msgs else "No explicit constraints found."

        # We provide a focused instruction set. The LLM must respect the language
        # you pass via ui_language (no hard-coded branching).
        prompt = f"""Role: Academic Advisor. Language: {ui_language}
You will answer the user's follow-up question about this specific academic major.
Use the CONTEXT and the eligibility analysis (computed from rules + Bagrut).
Do not invent requirements that aren't present in the rules.

CONTEXT:
{context}

ELIGIBILITY_ANALYSIS:
- STATUS: {elig_status}
- NOTES: {elig_notes}

USER_QUESTION:
{question}

INTENT: {intent}

GUIDELINES:
- Write clearly in {ui_language}.
- If intent is 'eligibility':
    • Start with a one-line verdict using STATUS.
    • Then list the key requirements and mention which are missing/unclear (from NOTES).
    • If gaps exist, say briefly how to close them (e.g., higher units/grade, psychometric, interview).
- If intent is 'courses':
    • Summarize the structure (core, labs/projects, electives) using what's in CONTEXT.
    • Include 3–6 example courses that appear relevant (from 'Sample courses' or 'Keywords').
- If intent is 'roles':
    • List common job paths and typical entry roles that align with the major’s focus (derive from keywords/courses).
- If 'general':
    • Give a short, helpful answer and suggest one next question the student could ask.
- Be concise (≤ 150 words).
- Never mention semesters or “current university studies”; the student is pre-admission.

Now write the answer only (no headers like CONTEXT or GUIDELINES)."""

        return self.llm.invoke(prompt).content.strip()
