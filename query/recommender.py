# query/recommender.py
from __future__ import annotations

"""
Top-k major recommendation using cosine similarity between a student's
50-criteria profile vector and each major's 50-criteria scores.
"""

import json
import math
from typing import List, Dict, Tuple

# Keep the SAME order as majors embeddings
try:
    from ingestion.majors_embeddings import get_criteria
except Exception as e:
    raise RuntimeError("Cannot import get_criteria() from ingestion.majors_embeddings. "
                       "Ensure the file exists and PYTHONPATH is correct.") from e


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-9
    nb = math.sqrt(sum(y*y for y in b)) or 1e-9
    return dot / (na * nb)


def _vector_from_scores(scores: Dict[str, int]) -> List[float]:
    keys = [k for k, _ in get_criteria()]
    return [float(int(scores.get(k, 0))) for k in keys]  # 0..5 floats


def _agreement_top_dims(ps: Dict[str, int], ms: Dict[str, int], top: int = 5) -> List[Tuple[str, float, int, int]]:
    """
    Returns top `top` criteria with the best agreement between person and major.
    agreement = 1 - |p-m|/5  (range 0..1)
    """
    items: List[Tuple[str, float, int, int]] = []
    for k, _ in get_criteria():
        pv, mv = int(ps.get(k, 0)), int(ms.get(k, 0))
        agree = 1.0 - (abs(pv - mv) / 5.0)
        items.append((k, agree, pv, mv))
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:max(1, top)]


def load_majors(majors_embeddings_json: str) -> List[Dict]:
    with open(majors_embeddings_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("majors_embeddings.json must be a JSON list.")
    return data


def recommend(person_json: str, majors_embeddings_json: str, k: int = 3) -> List[Dict]:
    """
    Load person profile JSON + majors embeddings JSON and return top-k matches.
    Each item includes similarity and a short 'why' (top aligned criteria).
    """
    with open(person_json, "r", encoding="utf-8") as f:
        person = json.load(f)
    pscores: Dict[str, int] = person.get("scores", {})
    pvec = _vector_from_scores(pscores)

    majors = load_majors(majors_embeddings_json)
    ranked: List[Dict] = []
    for m in majors:
        mscores: Dict[str, int] = m.get("scores", {})
        mvec = _vector_from_scores(mscores)
        sim = _cosine(pvec, mvec)
        why = _agreement_top_dims(pscores, mscores, top=5)
        ranked.append({
            "english_name": m.get("english_name") or m.get("original_name"),
            "original_name": m.get("original_name"),
            "similarity": round(sim, 4),
            "top_alignment": [
                {"criterion": c, "agreement": round(a, 3), "person": pv, "major": mv}
                for c, a, pv, mv in why
            ],
            "source": m.get("source", "")
        })
    ranked.sort(key=lambda x: x["similarity"], reverse=True)
    return ranked[:max(1, k)]


def rank_all(person_scores: Dict[str, int], majors_embeddings_json: str) -> List[Dict]:
    """
    Variant that accepts raw person scores (no file I/O) and returns full ranking.
    """
    pscores = {k: int(v) for k, v in person_scores.items()}
    pvec = _vector_from_scores(pscores)
    majors = load_majors(majors_embeddings_json)
    ranked: List[Dict] = []
    for m in majors:
        mscores: Dict[str, int] = m.get("scores", {})
        mvec = _vector_from_scores(mscores)
        sim = _cosine(pvec, mvec)
        ranked.append({
            "english_name": m.get("english_name") or m.get("original_name"),
            "original_name": m.get("original_name"),
            "similarity": round(sim, 4),
        })
    ranked.sort(key=lambda x: x["similarity"], reverse=True)
    return ranked
