# query/person_embeddings.py
from __future__ import annotations
"""
Stateless person embeddings (0..100) with default logging.
- Each update appends a snapshot to ./person_embeddings.jsonl
- Also writes the latest state to ./person_embeddings.json
- No persistent per-person profiles

Usage:
    from query.person_embeddings import PersonSession, update_person_session

    sess = PersonSession()  # starts at all zeros
    res = update_person_session(sess, "q1", "I love math and data.")
    print(res["vector"])
"""

import os, json, time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from embeddings import (
    get_criteria_keys, get_criteria, zero_scores,
    SCALE_MIN, SCALE_MAX, clamp_int, safe_json_loads
)

# ---- Defaults for files ----
DEFAULT_LOG_JSONL = "person_embeddings.jsonl"   # append-only history
DEFAULT_LATEST_JSON = "person_embeddings.json"  # latest snapshot only

# ---- LLM bits ----
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    StrOutputParser = None


# ------------------------- LLM helpers -------------------------

def _make_llm(model_env: str = "GEMINI_MODEL", default_model: str = "gemini-2.0-flash",
              temperature: float = 0.0):
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain_google_genai not installed.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    model_name = os.getenv(model_env, default_model)
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def _prompt_for_person():
    if ChatPromptTemplate is None:
        raise RuntimeError("langchain-core not installed.")
    rubric_lines = "\n".join([f"- {k}: {d}" for k, d in get_criteria()])
    keys = get_criteria_keys()
    template = (
        "You are evaluating a PERSON on the SAME rubric used for majors.\n"
        f"Return ONLY a JSON object mapping EACH criterion key to an INTEGER {SCALE_MIN}..{SCALE_MAX} (no missing keys).\n\n"
        "Rules:\n"
        f"- {SCALE_MIN} = not evident at all; {SCALE_MAX} = extremely strong evidence.\n"
        "- Provide ALL keys; integers only; no commentary; JSON only.\n\n"
        "Rubric keys and meanings:\n"
        f"{rubric_lines}\n\n"
        "PRIOR HISTORY (optional): {history}\n"
        "CURRENT ANSWER: {answer}\n\n"
        "JSON object with ONLY these keys (all must be present):\n"
        f"{keys}\n"
    )
    return ChatPromptTemplate.from_template(template)

def _score_person_once(answer_text: str, history_text: str = "", temperature: float = 0.0) -> Dict[str, int]:
    """Single LLM call → full rubric dict (0..100) for this answer."""
    llm = _make_llm(temperature=temperature)
    parser = StrOutputParser()
    prompt = _prompt_for_person()
    chain = prompt | llm | parser
    raw = chain.invoke({"history": history_text or "", "answer": answer_text or ""})
    obj = safe_json_loads(raw, default={})
    keys = get_criteria_keys()
    out = {k: clamp_int((obj.get("scores", obj) or {}).get(k, 0), SCALE_MIN, SCALE_MAX) for k in keys}
    return out


# ------------------------- Stateless session -------------------------

@dataclass
class PersonSession:
    """In-memory running vector; starts at all zeros (0..100)."""
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, int] = field(default_factory=zero_scores)   # 0..100, all zeros

    def vector(self) -> List[float]:
        keys = get_criteria_keys()
        return [float(clamp_int(self.scores.get(k, 0))) for k in keys]

    def apply_full_scores(self, full: Dict[str, int], weight: float = 1.0) -> None:
        """Running weighted average over answers (0..100)."""
        w = max(0.1, min(2.0, float(weight)))
        keys = get_criteria_keys()
        for k in keys:
            v = clamp_int(full.get(k, 0))
            self.sums[k] = float(self.sums.get(k, 0.0)) + v * w
            self.counts[k] = float(self.counts.get(k, 0.0)) + w
            avg = self.sums[k] / max(1e-9, self.counts[k])
            self.scores[k] = clamp_int(round(avg))

    # ---------- logging ----------
    def _append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

    def _write_latest_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            "criteria_keys": get_criteria_keys(),
            "scores": self.scores,
            "vector": self.vector(),
            "counts": self.counts,
            "sums": self.sums,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def update(self, question_id: str, answer_text: str, history_text: str = "",
               weight: float = 1.0, temperature: float = 0.0,
               log_jsonl: Optional[str] = None, latest_json: Optional[str] = None) -> Dict[str, Any]:
        """
        1) Score this answer on the rubric (0..100)
        2) Merge into running session
        3) Log to JSONL (append) and write latest JSON (overwrite)
        """
        full = _score_person_once(answer_text=answer_text, history_text=history_text, temperature=temperature)
        self.apply_full_scores(full, weight=weight)

        stamp = int(time.time())
        snapshot = {
            "timestamp": stamp,
            "question_id": question_id,
            "answer": answer_text,
            "weight": float(weight),
            "applied_scores": full,   # from this answer
            "scores": self.scores,    # after update
            "vector": self.vector(),  # after update
            "counts": self.counts,
            "sums": self.sums,
        }

        # default paths if not provided
        log_jsonl = log_jsonl or DEFAULT_LOG_JSONL
        latest_json = latest_json or DEFAULT_LATEST_JSON

        self._append_jsonl(log_jsonl, snapshot)
        self._write_latest_json(latest_json)

        return {"applied": full, "vector": self.vector(), "log_jsonl": log_jsonl, "latest_json": latest_json}


# ------------------------- Convenience API -------------------------

def update_person_session(session: PersonSession, question_id: str, answer_text: str,
                          history_text: str = "", weight: float = 1.0, temperature: float = 0.0,
                          log_jsonl: Optional[str] = None, latest_json: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper around PersonSession.update(...).
    If paths are omitted, defaults are:
      - ./person_embeddings.jsonl (append history)
      - ./person_embeddings.json  (current snapshot)
    """
    return session.update(
        question_id=question_id, answer_text=answer_text, history_text=history_text,
        weight=weight, temperature=temperature,
        log_jsonl=log_jsonl, latest_json=latest_json
    )
