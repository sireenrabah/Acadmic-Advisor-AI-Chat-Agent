# generation/scorer.py
from __future__ import annotations

import os, json, re
from typing import Dict, List, Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None


_NUM = re.compile(r"\b([0-9]{1,3})\b")


def _make_llm(temperature: float = 0.2):
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain_google_genai is not installed.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=float(temperature))


def _safe_int(x: str) -> Optional[int]:
    m = _NUM.search(x or "")
    if not m:
        return None
    v = int(m.group(1))
    if 0 <= v <= 100:
        return v
    return None


def score_answer_multi(
    *,
    criteria: List[str],
    answer_text: str,
    history_text: str = "",
    temperature: float = 0.35,
) -> Dict[str, float]:
    """
    Score an answer for EACH criterion in 'criteria' on 0..100.
    Returns dict {key: float score}. Missing keys are omitted.
    Pure scorer — no re-asking, no hints.
    """
    if not criteria:
        return {}

    if ChatGoogleGenerativeAI is None or ChatPromptTemplate is None:
        # fallback: neutral scores if LLM unavailable
        return {k: 60.0 for k in criteria}

    llm = _make_llm(temperature=temperature)

    # Escape literal braces for LangChain templating safety.
    sys = (
        "You are grading a student's short answer on multiple rubric criteria.\n"
        "For each criterion, output a single integer 0..100 (no explanation).\n"
        "Return STRICT JSON only, like this: "
        "{{\"communication\": 73, \"planning\": 61}}\n"
        "No extra text."
    )

    human = (
        "Criteria keys (score each; 0..100): {keys}\n\n"
        "Student answer:\n{ans}\n\n"
        "Recent conversation context (optional):\n{hist}\n\n"
        "Return STRICT JSON mapping each key to an integer 0..100.\n"
        "Example format only (do NOT copy keys):\n"
        "{{\n"
        "  \"<key>\": 0..100,\n"
        "  \"<key2>\": 0..100\n"
        "}}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        ("human", human),
    ])

    raw = (prompt | llm).invoke({
        "keys": ", ".join(criteria),
        "ans": answer_text or "",
        "hist": history_text or "",
    }).content.strip()

    # Try JSON first
    try:
        data = json.loads(raw)
        out: Dict[str, float] = {}
        for k in criteria:
            if k in data and isinstance(data[k], (int, float)):
                v = int(data[k])
                if 0 <= v <= 100:
                    out[k] = float(v)
        if out:
            return out
    except Exception:
        pass

    # Defensive fallback: single number → apply to all
    v = _safe_int(raw)
    if v is not None:
        return {k: float(v) for k in criteria}

    # Last resort neutral
    return {k: 60.0 for k in criteria}
