# embeddings.py
from __future__ import annotations
"""
Shared rubric + helpers for BOTH majors and people.
No prompt builders here — keep this file “prompt-free”.
"""

import os, re, json
from typing import Any, Dict, List, Tuple

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
    ("innovation_openness", "נכונות لنיסוי וטעייה, פתיחות לחדשנות."),
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

def zero_scores() -> Dict[str, int]:
    return {k: 0 for k in get_criteria_keys()}

# Scale is 0..100 everywhere
SCALE_MIN = 0
SCALE_MAX = 100

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
