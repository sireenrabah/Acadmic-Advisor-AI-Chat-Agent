# ingestion/majors_embeddings.py
from __future__ import annotations
"""
Majors rubric scorer (0..100) without EmbeddingsBase.
- Scores each major on the shared rubric using a majors-specific prompt
- OUTPUT ONLY: original_name, english_name, scores, vector, source
  (no keywords/sample_courses duplication; those stay in extracted_majors.json)
- Also writes a sidecar schema file <out>.schema.json
- Skips rebuild if out_json already exists with a non-empty list
"""

import os, json
from typing import List, Dict, Any
from dotenv import load_dotenv

from embeddings import (
    get_criteria, get_criteria_keys, get_criteria_map,
    SCALE_MIN, SCALE_MAX, clamp_int, safe_json_loads
)

# LLM bits
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    StrOutputParser = None

load_dotenv()


# ------------------------- helpers -------------------------

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

def _make_llm(model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain_google_genai is not installed.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)

def _prompt_for_major():
    if ChatPromptTemplate is None:
        raise RuntimeError("langchain-core is not installed.")
    rubric_lines = "\n".join([f"- {k}: {d}" for k, d in get_criteria()])
    keys = get_criteria_keys()
    template = (
        "You are evaluating an academic MAJOR on a fixed rubric.\n"
        f"Return ONLY a JSON object mapping EACH criterion key to an INTEGER {SCALE_MIN}..{SCALE_MAX} (no missing keys).\n\n"
        "Rules:\n"
        f"- {SCALE_MIN} = not present at all; {SCALE_MAX} = extremely strong emphasis.\n"
        "- Provide ALL keys; integers only; no commentary; JSON only.\n\n"
        "Rubric keys and meanings:\n"
        f"{rubric_lines}\n\n"
        "MAJOR NAME (English): {name}\n"
        "SAMPLE COURSES (from extracted majors): {courses}\n"
        "KEYWORDS (from extracted majors): {keywords}\n\n"
        "JSON object with ONLY these keys (all must be present):\n"
        f"{keys}\n"
    )
    return ChatPromptTemplate.from_template(template)

def _score_major_once(name: str, courses: List[str], keywords: List[str], temperature: float = 0.0) -> Dict[str, int]:
    """Single LLM call → full rubric dict (0..100) for this major."""
    llm = _make_llm(temperature=temperature)
    parser = StrOutputParser()
    prompt = _prompt_for_major()
    chain = prompt | llm | parser
    raw = chain.invoke({
        "name": name or "",
        "courses": json.dumps(courses or [], ensure_ascii=False),
        "keywords": json.dumps(keywords or [], ensure_ascii=False),
    })
    obj = safe_json_loads(raw, default={})
    keys = get_criteria_keys()
    src = obj.get("scores", obj) if isinstance(obj, dict) else {}
    return {k: clamp_int(src.get(k, 0), SCALE_MIN, SCALE_MAX) for k in keys}


# ------------------------- main builder -------------------------

def build_majors_embeddings(in_json: str = "extracted_majors.json",
                           out_json: str = "majors_embeddings.json",
                           model_name: str = "gemini-2.0-flash",
                           temperature: float = 0.0) -> str:
    """
    Score all majors on the shared rubric (0..100).
    INPUT: extracted_majors.json (contains keywords/sample_courses)
    OUTPUT: majors_embeddings.json with minimal fields (no duplication).
    Also writes <out_json>.schema.json describing the rubric.
    """
    if os.path.exists(out_json):
        n = _json_len(out_json)
        if n > 0:
            print(f"[skip] {out_json} already exists with {n} majors. Skipping build.")
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

    # Ensure prompt objects exist early
    _ = _prompt_for_major()

    out_items: List[Dict[str, Any]] = []
    keys = get_criteria_keys()

    for i, item in enumerate(majors, 1):
        name = item.get("english_name") or item.get("original_name") or ""
        courses = item.get("sample_courses", []) or []
        keywords = item.get("keywords", []) or []
        try:
            scores = _score_major_once(name=name, courses=courses, keywords=keywords, temperature=temperature)
        except Exception as e:
            print(f"[warn] Scoring failed for '{name}': {e}")
            scores = {k: 0 for k in keys}

        # Minimal output (no keywords/sample_courses duplication)
        out_items.append({
            "original_name": item.get("original_name", ""),
            "english_name": item.get("english_name", name),
            "scores": scores,
            "vector": [float(scores[k]) for k in keys],  # aligned to rubric keys
            "source": item.get("source", ""),
        })
        if i % 5 == 0:
            print(f"[embed] Scored {i}/{len(majors)} majors...")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    _write_schema_sidecar(out_json)

    print(f"[embed] Wrote {len(out_items)} majors -> {out_json}")
    return out_json
