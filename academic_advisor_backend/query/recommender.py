# query/recommender.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from embeddings.embeddings import get_criteria_keys
from embeddings.person_embeddings import PersonProfile
from query.bagrut_features import eligibility_flags_from_rules

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


@dataclass
class StepRec:
    english_name: str
    original_name: str
    score: float
    short_reason: str
    hint: Optional[str] = None


@dataclass
class FinalRec:
    english_name: str
    original_name: str
    score: float
    rationale: str
    bagrut_alignment_pct: int
    eligibility_summary: Dict[str, Any]


class Recommender:
    """
    Two-phase recommendations:
      - recommend_step(...)  -> chat-friendly suggestions (short reasons, no heavy eligibility)
      - recommend_final(...) -> Top-K with rationale + eligibility summary

    Notes:
      * Person vector is built lazily; if scores aren't set yet, we use neutral 50s.
      * Language is controlled externally via set_language().
    """

    def __init__(
        self,
        *,
        person: PersonProfile,
        majors: List[Dict[str, Any]],
        ui_language: str,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        w_cosine: float = 0.45,
        w_rubric_align: float = 0.35,
        w_bagrut_align: float = 0.20,
        eligibility_penalty: float = 0.10,
    ):
        self.person = person
        self.majors = majors or []
        self.ui_language = ui_language
        self.llm = llm
        self._keys = get_criteria_keys()
        # Build person vector lazily to avoid touching person.scores at init
        self._pvec: Optional[np.ndarray] = None

        self.w_cosine = float(w_cosine)
        self.w_rubric_align = float(w_rubric_align)
        self.w_bagrut_align = float(w_bagrut_align)
        self.eligibility_penalty = float(eligibility_penalty)

    # ---- language / binding -------------------------------------------------
    def set_language(self, lang: str):
        self.ui_language = lang or self.ui_language

    def bind(self, majors: List[Dict[str, Any]]):
        self.majors = majors or []

    # ---- internal: safe scores + lazy person vector -------------------------
    def _get_scores_dict(self) -> Dict[str, float]:
        # Prefer an explicit scores dict if someone attached one
        if hasattr(self.person, "scores") and isinstance(self.person.scores, dict):
            return self.person.scores

        # PersonProfile exposes as_dict()
        if hasattr(self.person, "as_dict") and callable(self.person.as_dict):
            try:
                d = self.person.as_dict()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass

        # Fallback: neutral 50s per key
        return {k: 50.0 for k in self._keys}


    def _ensure_person_vec(self) -> np.ndarray:
        if self._pvec is not None:
            return self._pvec
        scores = self._get_scores_dict()
        vals = [max(0.0, min(100.0, float(scores.get(k, 50.0)))) / 100.0 for k in self._keys]
        self._pvec = np.array(vals, dtype=float)
        return self._pvec

    # ---- similarity & blends -----------------------------------------------
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _rubric_align(self, v: np.ndarray) -> float:
        p = self._ensure_person_vec()
        d = np.abs(p - v).mean()
        return max(0.0, 1.0 - d)

    def _bagrut_align(self, v: np.ndarray) -> float:
        # Dampened reuse of rubric alignment to reflect Bagrut-informed tendencies
        ra = self._rubric_align(v)
        return math.sqrt(max(0.0, ra))

    def _blend(self, v: np.ndarray) -> float:
        p = self._ensure_person_vec()
        c = self._cosine(p, v)
        r = self._rubric_align(v)
        b = self._bagrut_align(v)
        return self.w_cosine * c + self.w_rubric_align * r + self.w_bagrut_align * b

    def _rank(self) -> List[Tuple[int, float]]:
        scored: List[Tuple[int, float]] = []
        p = self._ensure_person_vec()
        for i, m in enumerate(self.majors):
            vec = np.array(m.get("vector", []), dtype=float)
            if vec.size != p.size:
                continue
            scored.append((i, self._blend(vec)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    # ---- LLM helpers (language via prompt) ----------------------------------
    def _one_liner(self, m: Dict[str, Any]) -> str:
        if not self.llm:
            return f"{m.get('english_name') or m.get('original_name')}"
        prompt = f"""You are an academic advisor. Language: {self.ui_language}
Write ONE sentence (≤20 words) explaining why this major may fit the student.
Major: {m.get('english_name') or m.get('original_name')}
Keywords: {', '.join(m.get('keywords') or [])}
Sample courses: {', '.join(m.get('sample_courses') or [])}
Only output the sentence."""
        return self.llm.invoke(prompt).content.strip().split("\n")[0][:220]

    def _paragraph(self, m: Dict[str, Any], bagrut_pct: int, eligibility_hint: str) -> str:
        if not self.llm:
            return f"Alignment ≈ {bagrut_pct}%. {eligibility_hint}".strip()
        
        # CRITICAL: Enforce target language with example
        prompt = f"""You are an academic advisor. CRITICAL: Respond ONLY in {self.ui_language} language.

Write a concise 2-3 sentence explanation of why this major matches the student's profile.
- Mention their relevant strengths from Bagrut
- Keep it natural and encouraging
- Language: {self.ui_language}

Major: {m.get('original_name') or m.get('english_name')}
Keywords: {', '.join((m.get('keywords') or [])[:5])}
Sample courses: {', '.join((m.get('sample_courses') or [])[:3])}
Bagrut alignment: ~{bagrut_pct}%
Eligibility: {eligibility_hint}

Your explanation in {self.ui_language}:"""
        return self.llm.invoke(prompt).content.strip()

    # ---- public APIs --------------------------------------------------------
    def recommend_step(self, top_k: int = 2) -> List[Dict[str, Any]]:
        out: List[StepRec] = []
        for idx, s in self._rank()[:max(1, top_k)]:
            m = self.majors[idx]
            one = self._one_liner(m)
            vec = np.array(m.get("vector", []), dtype=float)
            hint = None
            if self._rubric_align(vec) < 0.7:
                hint = "Clarify what you like more: theory, hands-on labs, or projects."
            out.append(StepRec(
                english_name=m.get("english_name", ""),
                original_name=m.get("original_name", ""),
                score=round(float(s), 4),
                short_reason=one,
                hint=hint
            ))
        return [asdict(x) for x in out]

    def recommend_final(self, top_k: int = 3, bagrut_json: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # DEBUG: Show person vector state
        person_scores = self._get_scores_dict()
        top_person_scores = sorted(person_scores.items(), key=lambda x: -x[1])[:5]
        print(f"[recommender:debug] Person top 5 scores:")
        for crit, score in top_person_scores:
            print(f"  {crit}: {score:.1f}")
        
        prelim = self._rank()[:max(1, top_k * 2)]
        
        # DEBUG: Show top scored majors BEFORE eligibility
        print(f"[recommender:debug] Top {len(prelim)} majors before eligibility:")
        for idx, base in prelim[:5]:
            m = self.majors[idx]
            print(f"  {m.get('original_name', 'Unknown')}: {base:.3f}")
        
        adjusted: List[Tuple[int, float, Dict[str, Any]]] = []
        for idx, base in prelim:
            m = self.majors[idx]
            ok, msgs = eligibility_flags_from_rules(m.get("eligibility_rules") or {}, bagrut_json or {})
            penalty = self.eligibility_penalty if (not ok) else 0.0
            adjusted.append((idx, max(0.0, base - penalty), {"ok": ok, "msgs": msgs}))
        adjusted.sort(key=lambda t: t[1], reverse=True)

        results: List[FinalRec] = []
        for idx, s, el in adjusted[:max(1, top_k)]:
            m = self.majors[idx]
            vec = np.array(m.get("vector", []), dtype=float)
            bag_pct = int(round(self._rubric_align(vec) * 100))
            elig_hint = "Eligible." if el["ok"] else ("; ".join(el["msgs"][:4]) or "Eligibility unclear.")
            rationale = self._paragraph(m, bag_pct, elig_hint)
            results.append(FinalRec(
                english_name=m.get("english_name", ""),
                original_name=m.get("original_name", ""),
                score=round(float(s), 4),
                rationale=rationale,
                bagrut_alignment_pct=bag_pct,
                eligibility_summary={"pass": el["ok"], "notes": el["msgs"]}
            ))
        return [asdict(x) for x in results]
