# query/bagrut_features.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import json, os, math

# ---- Gemini-powered translators you already have ----
try:
    # MajorsTranslator is for batch name translation (we won't use it here)
    from translator import UITranslator
except Exception:
    UITranslator = None  # graceful fallback

# Single, lazy UI translator instance (0-temp)
_UI_TR = None
def _ui_tr():
    global _UI_TR
    if _UI_TR is None:
        if UITranslator is not None:
            _UI_TR = UITranslator()  # builds its own Gemini llm with temp=0
        else:
            class _Noop:
                def tr(self, lang: str, text: str) -> str: return text
            _UI_TR = _Noop()
    return _UI_TR

# ---------------------------------------------------------------------------------
# Subject → criteria bridge (MUST match embeddings.py CRITERIA keys!)
# Canonicalization uses Gemini via UITranslator: we translate any raw subject to
# English and then compare against these keys (case-insensitive).
#
# CRITICAL: These keys MUST exist in embeddings.py CRITERIA!
# Available keys: data_analysis, quantitative_reasoning, logical_problem_solving,
# pattern_recognition, systems_thinking, risk_decision, strategic_thinking,
# cognitive_flexibility, curiosity_interdisciplinary, creative_solutions,
# innovation_openness, opportunity_recognition, social_sensitivity, teamwork,
# communication_mediation, ethical_awareness, psychological_interest,
# theoretical_patience, attention_to_detail, leadership, hebrew_proficiency,
# english_proficiency, hebrew_expression, english_expression
# ---------------------------------------------------------------------------------
SUBJECT2CRITERIA: Dict[str, List[str]] = {
    # STEM subjects → analytical/quantitative/logical
    "Mathematics": [
        "quantitative_reasoning",      # חשיבה כמותית
        "logical_problem_solving",     # פתרון בעיות לוגיות
        "pattern_recognition"          # זיהוי דפוסים
    ],
    "Physics": [
        "quantitative_reasoning",
        "systems_thinking",            # הבנת מערכות פיזיקליות
        "logical_problem_solving"
    ],
    "Computer Science": [
        "logical_problem_solving",     # אלגוריתמיקה
        "data_analysis",               # מבני נתונים
        "systems_thinking",            # ארכיטקטורה
        "pattern_recognition"
    ],
    "Chemistry": [
        "data_analysis",               # ניתוח ניסויים
        "attention_to_detail",         # דיוק במעבדה
        "quantitative_reasoning"
    ],
    "Biology": [
        "data_analysis",               # מחקר ביולוגי
        "attention_to_detail",         # עבודת מעבדה
        "ethical_awareness",           # דילמות ביו-אתיות
        "systems_thinking"
    ],
    
    # Language subjects → communication/expression
    "English": [
        "english_proficiency",         # הבנה אקדמית
        "english_expression",          # התבטאות כתובה/דיבור
        "communication_mediation"      # תקשורת
    ],
    "Hebrew": [
        "hebrew_proficiency",
        "hebrew_expression",
        "communication_mediation"
    ],
    "Arabic": [
        "hebrew_proficiency",          # ערבית לערבים = רמה גבוהה בעברית
        "communication_mediation",
        "social_sensitivity"           # הבנה רב-תרבותית
    ],
    
    # Humanities → theoretical/contextual thinking
    "History": [
        "theoretical_patience",        # למידה תאורטית
        "attention_to_detail",         # ניתוח מקורות
        "curiosity_interdisciplinary", # הקשרים היסטוריים
        "communication_mediation"
    ],
    "Civics": [
        "ethical_awareness",           # דילמות מוסריות
        "social_sensitivity",          # מודעות חברתית
        "communication_mediation",
        "risk_decision"                # החלטות מדיניות
    ],
    "Geography": [
        "systems_thinking",            # מערכות סביבתיות
        "data_analysis",               # נתונים גיאוגרפיים
        "pattern_recognition"          # מגמות מרחביות
    ],
    
    # Technical subjects
    "Electronics and Computers": [
        "logical_problem_solving",
        "systems_thinking",
        "attention_to_detail"
    ],
    
    # Religious/cultural subjects
    "Islamic Heritage and Religion": [
        "ethical_awareness",
        "social_sensitivity",
        "theoretical_patience"
    ],
}

# helper: try to match variants like "History for Arabs" -> "History"
_CANON_FALLBACKS = {
    "history for arabs": "History",
    "arabic for arabs": "Arabic",
    "islamic heritage": "Islamic Heritage and Religion",
    "electronics": "Electronics and Computers",
    "electronics and computers": "Electronics and Computers",
}

def _grade_to_0_100(g: float) -> float:
    try:
        return max(0.0, min(100.0, float(g)))
    except Exception:
        return 0.0

def _units_weight(u: int) -> float:
    try:
        u = int(u)
    except Exception:
        u = 0
    # gentle weighting
    if u >= 5: return 1.0
    if u == 4: return 0.9
    if u == 3: return 0.8
    if u == 2: return 0.7
    if u == 1: return 0.6
    return 0.5

def _guess_units_from_name_heuristic(en_name: str) -> int:
    """
    When the extractor couldn't get 'units', pick a reasonable neutral default.
    We avoid country-specific hard rules; just a light heuristic:
      - STEM often 4–5, humanities/civics often 2–3.
    """
    t = (en_name or "").lower()
    if any(k in t for k in ["math", "physic", "computer", "chem", "bio", "electronic"]):
        return 5
    if any(k in t for k in ["english", "hebrew", "arabic"]):
        return 5
    if any(k in t for k in ["history", "civic", "heritage", "religion", "culture"]):
        return 2
    return 3

def to_canonical_subject_en(raw_subject: str) -> str:
    """
    Use your Gemini-backed UI translator to map a raw subject (any language)
    into a canonical English label. Then, coerce to our SUBJECT2CRITERIA keys
    when reasonable; otherwise return the translated form (still used for display).
    """
    if not raw_subject:
        return ""
    # translate raw -> English (UITranslator picks English when lang is 'en')
    # we force 'en' by passing lang='en'
    tr = _ui_tr()
    en = tr.tr("en", str(raw_subject).strip())
    base = en.strip()

    # Try exact key match (case-insensitive)
    for k in SUBJECT2CRITERIA.keys():
        if base.lower() == k.lower():
            return k

    # Try simple fallbacks
    fb = _CANON_FALLBACKS.get(base.lower())
    if fb:
        return fb

    # As a last resort, Title Case the translated string (won't add criteria, but ok)
    return base.title()

# ---------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------
def load_bagrut(path: str = "state/extracted_bagrut.json") -> Dict:
    if not os.path.exists(path):
        return {"by_subject": {}, "total_units": 0, "weighted_average": None}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both shapes:
    # 1) {"by_subject": {...}}
    # 2) {"subjects": [{"subject": "...", "final_grade": ..., "units": ..., "year": ...}, ...]}
    if isinstance(data, dict) and "subjects" in data and "by_subject" not in data:
        by_subj: Dict[str, Dict] = {}
        for row in data.get("subjects") or []:
            raw = (row or {}).get("subject") or ""
            if not raw:
                continue
            en = to_canonical_subject_en(raw)  # normalize with Gemini
            g = _grade_to_0_100((row or {}).get("final_grade") or 0)
            u = (row or {}).get("units")
            try:
                u = int(u) if u is not None else None
            except Exception:
                u = None
            if u is None:
                u = _guess_units_from_name_heuristic(en)
            by_subj[en] = {"final_grade": g, "units": u}
        data = {"by_subject": by_subj}

    return data

# ---------- normalize bagrut JSON ----------
def normalize_bagrut(bagrut: Dict) -> Dict:
    """Ensure fields exist and compute weighted_average if missing."""
    if not isinstance(bagrut, dict):
        return {"by_subject": {}, "total_units": 0, "weighted_average": None}

    # If it arrived as {"subjects":[...]}, route through load/save shape:
    if "subjects" in bagrut and "by_subject" not in bagrut:
        bagrut = load_bagrut_data(bagrut)

    by_subject: Dict[str, Dict] = dict(bagrut.get("by_subject") or {})
    total_units = 0
    wsum = 0.0

    # Canonicalize + fill units
    new_by: Dict[str, Dict] = {}
    for subj_raw, rec in by_subject.items():
        en = to_canonical_subject_en(subj_raw)
        if not isinstance(rec, dict):
            continue
        u = rec.get("units")
        try:
            u = int(u) if u is not None else None
        except Exception:
            u = None
        if u is None:
            u = _guess_units_from_name_heuristic(en)

        g = _grade_to_0_100(rec.get("final_grade") or rec.get("grade") or 0)

        new_by[en] = {"units": u, "final_grade": g}
        total_units += u
        wsum += g * u

    avg = None
    if total_units > 0:
        avg = round(wsum / float(total_units), 2)

    out = {
        "by_subject": new_by,
        "total_units": total_units,
        "weighted_average": bagrut.get("weighted_average") if isinstance(bagrut.get("weighted_average"), (int, float)) else avg,
    }
    # helpful log for your console
    try:
        print(f"[debug] normalized subjects: {len(new_by)}, avg={out['weighted_average']}")
    except Exception:
        pass
    return out

def load_bagrut_data(data_in: Dict) -> Dict:
    """
    Same as load_bagrut(file) but accepts the parsed json object (shape with 'subjects')
    and returns {'by_subject': ...}.
    """
    by_subj: Dict[str, Dict] = {}
    for row in (data_in or {}).get("subjects") or []:
        raw = (row or {}).get("subject") or ""
        if not raw:
            continue
        en = to_canonical_subject_en(raw)
        # Support both "final_grade" and "grade" fields
        g = _grade_to_0_100((row or {}).get("final_grade") or (row or {}).get("grade") or 0)
        u = (row or {}).get("units")
        try:
            u = int(u) if u is not None else None
        except Exception:
            u = None
        if u is None:
            u = _guess_units_from_name_heuristic(en)
        by_subj[en] = {"final_grade": g, "units": u}
    return {"by_subject": by_subj}

# ---------- signals ----------
def bagrut_signals(bagrut: Dict) -> Dict[str, float]:
    """
    Aggregate per-criterion signals from canonicalized subjects.
    Unrecognized subjects are silently ignored.
    """
    bagrut = normalize_bagrut(bagrut)
    out: Dict[str, List[float]] = {}
    by_subject: Dict[str, Dict] = bagrut.get("by_subject", {})
    
    # DEBUG: Show what subjects we're processing
    print(f"[bagrut_signals:debug] Processing {len(by_subject)} subjects:")
    
    for subj, rec in by_subject.items():
        # subj is already canonical English
        grade = _grade_to_0_100(rec.get("final_grade") or 0)
        units = int(rec.get("units") or 0)
        w = _units_weight(units)
        val = grade * w
        
        crits = SUBJECT2CRITERIA.get(subj)
        if not crits:
            # try forgiving match by substring
            s = subj.lower()
            for k, v in SUBJECT2CRITERIA.items():
                if k.lower() in s or s in k.lower():
                    crits = v
                    break
        
        # DEBUG: Show mapping
        if crits:
            print(f"  ✓ {subj}: grade={grade:.0f}, units={units}, weight={w:.2f}, val={val:.1f} → {crits}")
        else:
            print(f"  ✗ {subj}: grade={grade:.0f}, units={units} → NOT MAPPED (no criteria)")
            continue
            
        for c in crits:
            out.setdefault(c, []).append(val)

    # aggregate + light optimism blend
    agg = {k: (sum(v) / len(v)) for k, v in out.items() if v}
    result = {k: 0.7 * s + 0.3 * min(100.0, s + 15.0) for k, s in agg.items()}
    
    # DEBUG: Show top aggregated signals
    top_signals = sorted(result.items(), key=lambda x: -x[1])[:5]
    print(f"[bagrut_signals:debug] Top 5 aggregated signals:")
    for crit, score in top_signals:
        print(f"  {crit}: {score:.1f}")
    
    return result

def seed_person_vector(criteria_keys: List[str], signals: Dict[str, float]) -> List[float]:
    return [float(signals.get(k, 50.0)) for k in criteria_keys]


# ADD THIS to query/bagrut_features.py
from typing import Dict, List, Tuple  # (keep if already imported)

def eligibility_flags_from_rules(eligibility_rules: Dict, bagrut: Dict) -> Tuple[bool, List[str]]:
    """
    Compare the student's (normalized) Bagrut against a program's eligibility_rules.
    Returns: (meets_requirements, messages)
      eligibility_rules shape (all fields optional):
        {
          "subjects": {
            "Mathematics": {"min_units": 5, "min_grade": 85},
            "English": {"min_units": 5, "min_grade": 90}
          },
          "bagrut_avg_min": 90,
          "psychometric_min": 620,          # informational (not in Bagrut)
          "english_level_min": "B2",        # informational
          "interview_required": true,       # informational
          "notes": ["..."]
        }
    """
    msgs: List[str] = []
    ok = True

    # ensure canonical EN subjects + weighted average available
    bnorm = normalize_bagrut(bagrut)
    have = bnorm.get("by_subject", {}) or {}

    # subject thresholds
    subj_rules: Dict[str, Dict] = (eligibility_rules or {}).get("subjects", {}) or {}
    for subj, req in subj_rules.items():
        if not isinstance(req, dict):
            continue
        # subjects in 'have' are already canonicalized to English
        s = have.get(subj) or have.get(subj.title()) or have.get(subj.strip())
        if not s:
            ok = False
            msgs.append(f"Missing {subj}.")
            continue
        try:
            u = int(s.get("units") or 0)
        except Exception:
            u = 0
        try:
            g = float(s.get("final_grade") or 0)
        except Exception:
            g = 0.0

        mu = req.get("min_units")
        mg = req.get("min_grade")

        if isinstance(mu, int) and u < mu:
            ok = False
            msgs.append(f"{subj} units {u} < {mu}.")
        if isinstance(mg, (int, float)) and g < float(mg):
            ok = False
            msgs.append(f"{subj} grade {int(g)} < {int(mg)}.")

    # overall bagrut average
    avg_req = (eligibility_rules or {}).get("bagrut_avg_min")
    if isinstance(avg_req, (int, float)):
        avg_have = bnorm.get("weighted_average")
        if isinstance(avg_have, (int, float)):
            if float(avg_have) < float(avg_req):
                ok = False
                msgs.append(f"Bagrut average {avg_have} < {avg_req}.")
        else:
            msgs.append("Bagrut average missing; cannot verify threshold.")

    # informational flags (not enforced against Bagrut JSON)
    psy_req = (eligibility_rules or {}).get("psychometric_min")
    if isinstance(psy_req, (int, float)):
        msgs.append(f"Psychometric minimum {int(psy_req)} (to be verified when provided).")

    el = (eligibility_rules or {}).get("english_level_min")
    if isinstance(el, str) and el.strip():
        msgs.append(f"Minimum English level: {el.strip()}.")

    if (eligibility_rules or {}).get("interview_required") is True:
        msgs.append("Interview may be required.")

    for n in (eligibility_rules or {}).get("notes") or []:
        if isinstance(n, str) and n.strip():
            msgs.append(n.strip())

    return ok, msgs
