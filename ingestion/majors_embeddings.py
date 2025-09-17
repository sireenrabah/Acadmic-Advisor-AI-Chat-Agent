# ingestion/majors_embeddings.py
from __future__ import annotations

"""
Majors embeddings builder using the shared EmbeddingsBase and shared CRITERIA.
Also writes a sidecar schema file with the criteria used.
Skips rebuild if out_json already exists and is non-empty.
"""

import os, json
from typing import List, Dict, Any
from dotenv import load_dotenv

from embeddings import EmbeddingsBase, get_criteria, get_criteria_keys, get_criteria_map

load_dotenv()

def _json_len(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0

def _write_schema_sidecar(out_json: str) -> None:
    schema_path = out_json.replace(".json", ".schema.json")
    payload = {
        "criteria_keys": get_criteria_keys(),
        "criteria": get_criteria_map()
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def build_major_embeddings(in_json: str = "extracted_majors.json",
                           out_json: str = "majors_embeddings.json") -> str:
    """
    Build embeddings for all majors by scoring each on the shared rubric.
    Skips rebuilding if `out_json` already exists and contains a non-empty list.
    Additionally writes `<out_json>.schema.json` with the criteria for reproducibility.
    """
    if os.path.exists(out_json):
        n = _json_len(out_json)
        if n > 0:
            print(f"[skip] {out_json} already exists with {n} majors. Skipping embeddings build.")
            # ensure schema sidecar exists
            _write_schema_sidecar(out_json)
            return out_json
        else:
            print(f"[warn] {out_json} exists but is empty/invalid. Rebuilding...")

    if not os.path.exists(in_json):
        raise FileNotFoundError(f"Input JSON not found: {in_json}")

    with open(in_json, "r", encoding="utf-8") as f:
        majors = json.load(f)
    if not isinstance(majors, list):
        raise ValueError("Input JSON must be a list of majors.")

    scorer = EmbeddingsBase(temperature=0.0)

    out_items: List[Dict[str, Any]] = []
    for i, item in enumerate(majors, 1):
        name = item.get("english_name") or item.get("original_name") or ""
        courses = item.get("sample_courses", [])
        keywords = item.get("keywords", [])
        try:
            scores = scorer.score_major(name=name, courses=courses, keywords=keywords)
        except Exception as e:
            print(f"[warn] Scoring failed for '{name}' :: {e}")
            scores = {k: 0 for k in get_criteria_keys()}

        out_items.append({
            "original_name": item.get("original_name", ""),
            "english_name": item.get("english_name", ""),
            "scores": scores,
            "vector": [float(scores[k]) for k in get_criteria_keys()],  # explicit vector aligned to keys
            "source": item.get("source", ""),
        })
        if i % 5 == 0:
            print(f"[embed] Scored {i}/{len(majors)} majors...")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    _write_schema_sidecar(out_json)

    print(f"[embed] Wrote {len(out_items)} majors -> {out_json}")
    return out_json
