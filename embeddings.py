# embeddings.py
from __future__ import annotations

"""
EmbeddingsBase — shared superclass + shared CRITERIA for BOTH majors and people.
Outputs are on a 0..5 integer scale by default.
"""

import os, re, json
from typing import Any, Dict, List, Tuple

# --------------------------- SHARED CRITERIA ---------------------------
# Single source of truth. Update here; both majors & person import from this file.
CRITERIA: List[Tuple[str, str]] = [
    ("data_analysis", "יכולת ניתוח נתונים והסקת מסקנות."),
    ("quantitative_reasoning", "חשיבה כמותית ושימוש במודלים מתמטיים."),
    ("logical_problem_solving", "פתרון בעיות לוגיות מורכבות."),
    ("pattern_recognition", "זיהוי דפוסים ומגמות במידע."),
    ("systems_thinking", "יכולת ראיית תמונה כוללת והבנת מערכות."),
    ("risk_decision", "שקלול סיכונים וקבלת החלטות בתנאי אי־ודאות."),
    ("strategic_thinking", "חשיבה אסטרטגית ארוכת טווח."),
    ("cognitive_flexibility", "גמישות מחשבתית והתמודדות עם שינויים."),
    ("curiosity_interdisciplinary", "סקרנות ויכולת לחבר בין תחומים שונים."),
    ("creative_solutions", "יכולת יצירת פתרונות מקוריים."),
    ("innovation_openness", "נכונות לניסוי וטעייה, פתיחות לחדשנות."),
    ("opportunity_recognition", "ראיית הזדמנויות והפיכתן לפעולה."),
    ("social_sensitivity", "רגישות למצבים חברתיים והבנת נקודות מבט שונות."),
    ("teamwork", "עבודה בצוות, שיתוף פעולה ויכולת למידה משותפת."),
    ("communication_mediation", "יכולת תקשורת בין־אישית וגישור."),
    ("ethical_awareness", "מודעות לאתיקה ודילמות ערכיות."),
    ("psychological_interest", "עניין במניעים ותהליכים פסיכולוגיים."),
    ("theoretical_patience", "סבלנות ללמידה תאורטית לצד יישום."),
    ("attention_to_detail", "קשב לדקויות בתקשורת ובמידע."),
    ("leadership", "לקיחת אחריות, הובלה וניהול יוזמות."),
    ("hebrew_proficiency", "ידע והבנה בעברית אקדמית/מקצועית."),
    ("english_proficiency", "ידע והבנה באנגלית אקדמית/מקצועית."),
    ("hebrew_expression", "יכולת התבטאות בעברית בכתב ובעל פה."),
    ("english_expression", "יכולת התבטאות באנגלית בכתב ובעל פה."),
]

def get_criteria() -> List[Tuple[str, str]]:
    return list(CRITERIA)

def get_criteria_keys() -> List[str]:
    return [k for k, _ in CRITERIA]

def get_criteria_map() -> Dict[str, str]:
    return {k: d for k, d in CRITERIA}


# --------------------------- LLM infra ---------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    StrOutputParser = None

SCALE_MIN = 0
SCALE_MAX = 100  # set to 100 if you want 0..100 rubric throughout


def clamp_int(x: Any, lo: int = SCALE_MIN, hi: int = SCALE_MAX) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def safe_json_loads(s: str, default: Any = None) -> Any:
    if not s:
        return default
    s = re.sub(r"^```(?:json)?", "", s.strip())
    s = re.sub(r"```$", "", s.strip())
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        s2 = s2.replace("'", '"')
        try:
            return json.loads(s2)
        except Exception:
            return default


class EmbeddingsBase:
    """
    Shared scorer for majors and people — builds prompts, calls LLM, coerces full dict.
    """
    def __init__(self,
                 criteria: List[Tuple[str, str]] = None,
                 model_env: str = "GEMINI_MODEL",
                 default_model: str = "gemini-2.0-flash",
                 temperature: float = 0.0):
        self.criteria = list(criteria or CRITERIA)
        self.keys = [k for k, _ in self.criteria]
        self.rubric_lines = "\n".join([f"- {k}: {desc}" for k, desc in self.criteria])
        self.model_env = model_env
        self.default_model = default_model
        self.temperature = temperature

    def _make_llm(self):
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain_google_genai not installed.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        model_name = os.getenv(self.model_env, self.default_model)
        return ChatGoogleGenerativeAI(model=model_name, temperature=self.temperature)

    def prompt_for_major(self):
        if ChatPromptTemplate is None:
            raise RuntimeError("langchain-core not installed.")
        template = (
            "You are evaluating an academic MAJOR on a rubric.\n"
            f"Return ONLY a JSON object mapping EACH criterion key to an INTEGER {SCALE_MIN}..{SCALE_MAX} (no missing keys).\n\n"
            "Rules:\n"
            f"- {SCALE_MIN} = not present at all; {SCALE_MAX} = extremely strong emphasis.\n"
            "- Provide ALL keys; integers only; no commentary; JSON only.\n\n"
            "Rubric keys and meanings:\n"
            f"{self.rubric_lines}\n\n"
            "MAJOR NAME (English): {name}\n"
            "SAMPLE COURSES: {courses}\n"
            "KEYWORDS: {keywords}\n\n"
            "JSON object with ONLY these keys (all must be present):\n"
            f"{self.keys}\n"
        )
        return ChatPromptTemplate.from_template(template)


    def _invoke(self, prompt, variables: Dict[str, Any]) -> Dict[str, int]:
        llm = self._make_llm()
        parser = StrOutputParser()
        chain = prompt | llm | parser
        raw = chain.invoke(variables)
        obj = safe_json_loads(raw, default={})
        return self._coerce_full_scores(obj)

    def _coerce_full_scores(self, obj: Any) -> Dict[str, int]:
        scores = obj.get("scores", obj) if isinstance(obj, dict) else {}
        out: Dict[str, int] = {}
        for k in self.keys:
            out[k] = clamp_int(scores.get(k, SCALE_MIN), SCALE_MIN, SCALE_MAX)
        return out

    def score_major(self, name: str, courses: List[str], keywords: List[str]) -> Dict[str, int]:
        prompt = self.prompt_for_major()
        return self._invoke(prompt, {
            "name": name or "",
            "courses": json.dumps(courses or [], ensure_ascii=False),
            "keywords": json.dumps(keywords or [], ensure_ascii=False),
        })

    def score_person(self, answer_text: str, history_text: str = "") -> Dict[str, int]:
        prompt = self.prompt_for_person()
        return self._invoke(prompt, {
            "history": history_text or "",
            "answer": answer_text or "",
        })
