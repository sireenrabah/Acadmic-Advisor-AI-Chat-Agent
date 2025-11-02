# person_embeddings.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json, os, re

# Bagrut seed utilities
from query.bagrut_features import load_bagrut, bagrut_signals, seed_person_vector

# Optional external scorer (your own implementation in generation/scorer.py)
try:
    # Expected to expose something like: score_answer_multi(text: str, criteria: List[str]) -> Dict[str, float]
    from generation.scorer import score_answer_multi  # type: ignore
except Exception:
    score_answer_multi = None  # gracefully degrade if not available


class PersonProfile:
    """
    Tracks the student's rubric-aligned vector (0–100 per criterion) and confidence per key.
    Seeds from Bagrut if available, then updates with minimal interview nudges.
    """

    def __init__(self, criteria_keys: List[str], init_vector: Optional[List[float]] = None, llm=None):
        self.criteria_keys = criteria_keys
        self.llm = llm  # Store LLM for potential scoring enhancements

        # ---- Seed vector ----
        # Priority: explicit init_vector > Bagrut seed > neutral(50)
        vector_from_arg = init_vector is not None and len(init_vector) == len(criteria_keys)

        bagrut_sig: Dict[str, Any] = {}
        if vector_from_arg:
            self.vector = [float(x) for x in init_vector]  # already aligned to criteria_keys
            self._seed_source = "arg"
        else:
            try:
                bjson = load_bagrut()              # safe if file missing in your implementation
                bagrut_sig = bagrut_signals(bjson) # {} if nothing useful
                self.vector = seed_person_vector(criteria_keys, bagrut_sig)
                self._seed_source = "bagrut" if bagrut_sig else "neutral"
            except Exception:
                self.vector = [50.0] * len(criteria_keys)
                self._seed_source = "neutral"

        # Confidence:
        # - If we seeded from Bagrut → medium-high confidence globally (0.6)
        # - Else → low baseline (0.3)
        base_conf = 0.6 if self._seed_source == "bagrut" else 0.3
        self.confidence: List[float] = [base_conf] * len(criteria_keys)

        # Keep signals if you want explanations elsewhere (optional)
        self._bagrut_signals: Dict[str, Any] = bagrut_sig

        # Turn-by-turn log
        self.history: List[Dict[str, Any]] = []  # {"key","delta","new_score","note",...}

    # --- Convenience accessors ---
    def as_dict(self) -> Dict[str, float]:
        """Return {criterion_key: score_0_100} aligned to rubric keys."""
        return {k: float(self.vector[i]) for i, k in enumerate(self.criteria_keys)}

    def save_snapshot(self, path: str = "person_embeddings.json") -> None:
        data = {
            "seed_source": self._seed_source,
            "criteria_keys": self.criteria_keys,
            "scores": self.as_dict(),
            "confidence": self.confidence,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def append_log(self, path: str = "person_embeddings.jsonl") -> None:
        """Append a JSONL line with the current state (useful per interview turn)."""
        rec = {
            "seed_source": self._seed_source,
            "scores": self.as_dict(),
            "confidence": self.confidence,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Updating with interview answers ---
    def update_from_answer(
        self,
        key: str,
        new_score_0_100: float,
        *,
        note: str = "",
        ext_weight: float = 1.0,
        temp: float = 0.0,
    ) -> None:
        """
        Blend the new answer into the vector and raise confidence for that key.
        EMA-style update; answers nudge rather than dominate the Bagrut seed.

        ext_weight ∈ [0.2, 1.0] scales how much a single turn moves the score.
        temp ∈ [0.0, 1.0] slightly scales how much confidence increases (higher temp => smaller bump).
        """
        if key not in self.criteria_keys:
            return

        i = self.criteria_keys.index(key)

        # Clamp inputs
        new_score = float(max(0.0, min(100.0, new_score_0_100)))
        conf = float(max(0.0, min(1.0, self.confidence[i])))
        ext_w = max(0.2, min(1.0, float(ext_weight)))
        temp = max(0.0, min(1.0, float(temp)))

        # EMA weight: lower confidence → bigger step; scale by external weight
        # Keep steps modest so Bagrut remains the anchor.
        base_w = 0.35 + 0.40 * (1.0 - conf)  # range ~[0.35, 0.75)
        w = base_w * ext_w

        old = self.vector[i]
        self.vector[i] = (1.0 - w) * old + w * new_score

        # Confidence bump: smaller if temperature is high or ext_weight is small
        bump = 0.12 * ext_w * (1.0 - 0.5 * temp)
        self.confidence[i] = min(0.95, conf + bump)

        self.history.append({
            "key": key,
            "old": round(old, 2),
            "incoming": round(new_score, 2),
            "new": round(self.vector[i], 2),
            "w": round(w, 3),
            "conf_before": round(conf, 3),
            "conf_after": round(self.confidence[i], 3),
            "note": note,
            "ext_weight": round(ext_w, 3),
            "temp": round(temp, 3),
        })

    def batch_update(self, updates: Dict[str, float], *, ext_weight: float = 1.0, temp: float = 0.0) -> None:
        """Update many keys at once (e.g., from a multi-criterion answer)."""
        for k, v in updates.items():
            self.update_from_answer(k, v, note="batch", ext_weight=ext_weight, temp=temp)

    # --- Utility ---
    @staticmethod
    def load_or_new(criteria_keys: List[str], path: str = "person_embeddings.json") -> "PersonProfile":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scores = data.get("scores", {})
                vec = [float(scores.get(k, 50.0)) for k in criteria_keys]
                prof = PersonProfile(criteria_keys, init_vector=vec)
                prof.confidence = data.get("confidence", [0.4]*len(criteria_keys))
                prof._seed_source = data.get("seed_source", "loaded")
                return prof
            except Exception:
                pass
        return PersonProfile(criteria_keys)


# --------- Simple parsing helpers (no LLM here) ---------
_NUM_0_100 = re.compile(r"\b([0-9]{1,3})\b")

def _extract_numeric_self_rating(text: str) -> Optional[float]:
    """Return the first integer 0..100 found in the text, else None."""
    for m in _NUM_0_100.finditer(text or ""):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return float(v)
    return None


def update_person_profile(
    profile: "PersonProfile",
    *,
    question_id: str,
    answer_text: str,
    history_text: str = "",
    weight: float = 1.0,
    temperature: float = 0.0,
) -> None:
    """
    Update PersonProfile from a single interview answer.

    Scoring order:
      1) If the user typed a 0–100 number → use it (explicit self-rating).
      2) If LLM available in profile → use it to score answer contextually
      3) Else, if available, call your external scorer (generation/scorer.py) to get a 0..100 for `question_id`.
      4) Else, use a tiny heuristic (positive/negative words).

    Then blend via EMA in PersonProfile.update_from_answer(...).
    """

    # 1) Strict numeric self-rating
    rating = _extract_numeric_self_rating(answer_text)

    # 2) LLM scorer (NEW: better than keywords)
    if rating is None and hasattr(profile, 'llm') and profile.llm:
        try:
            score_prompt = f"""Analyze this student answer and rate their interest/aptitude for: {question_id}

Answer: "{answer_text}"

Rate 0-100 where:
- 0-30: No interest / weak aptitude
- 40-60: Moderate / uncertain
- 70-100: Strong interest / high aptitude

Return only the number (0-100):"""
            
            response = profile.llm.invoke(score_prompt).content.strip()
            # Extract first number found
            import re
            match = re.search(r'\b(\d+)\b', response)
            if match:
                rating = float(match.group(1))
                rating = max(0.0, min(100.0, rating))
        except Exception:
            rating = None  # fall through to next method

    # 3) External scorer (your scorer class/function), if available
    if rating is None and callable(score_answer_multi):
        try:
            # We assume the scorer can return a dict {criterion_key: score}
            scored = score_answer_multi(answer_text or "", criteria=[question_id]) or {}
            if question_id in scored and isinstance(scored[question_id], (int, float)):
                rating = float(scored[question_id])
        except Exception:
            rating = None  # fall through to heuristic

    # 4) Fallback heuristic
    if rating is None:
        t = (answer_text or "").lower()
        pos = any(w in t for w in ["yes", "yep", "definitely", "love", "enjoy", "strong", "confident", "sure", "interested", "like", "good", "great", "prefer", "high"])
        neg = any(w in t for w in ["no", "not", "hard", "struggle", "weak", "unsure", "doubt", "dislike", "hate", "bad", "poor", "low"])
        if pos and not neg:
            rating = 70.0
        elif neg and not pos:
            rating = 40.0
        else:
            rating = 60.0

    # Apply EMA update with weight & temperature influence
    profile.update_from_answer(
        question_id,
        float(max(0.0, min(100.0, rating))),
        note="interview",
        ext_weight=float(weight),
        temp=float(temperature),
    )
