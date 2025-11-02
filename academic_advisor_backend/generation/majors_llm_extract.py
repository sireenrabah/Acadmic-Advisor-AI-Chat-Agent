from __future__ import annotations
import os, re, json
from typing import List, Dict, Any
from dotenv import load_dotenv

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None


_PROMPT = r"""
You are a precise data extractor for Israeli university catalogs (in Hebrew).
Return ONLY JSON — a list of **actual degree programs** (majors).

## Your task
From the given text, detect *study programs* offered by the institution.
A valid program is a line that clearly names a degree, e.g.:
- "תואר ראשון B.A. ...", "תואר ראשון B.Sc. ...", "תואר שני M.A. ...", etc.

Ignore:
- Marketing lines ("אם השבתם בחיוב", "ההזדמנות שלך", "במסגרת התואר...")
- Paragraphs not naming a degree
- Bullet course lists not linked to a specific program

## For each real program, extract:
{
  "original_name": "exact Hebrew title of the degree",
  "sample_courses": [up to 20 course titles, short lines only],
  "eligibility_raw": "verbatim admission conditions paragraph (if any)",
  "eligibility_rules": {
      "psychometric_min": int?,
      "bagrut_avg_min": int?,
      "subjects": {
          "Mathematics"?: {"min_units"?: int, "min_grade"?: int},
          "English"?: {"min_units"?: int, "min_grade"?: int}
      },
      "english_level_min": str?,
      "interview_required": bool?,
      "notes": [string]
  }
}

## Notes
- Do not invent data.
- "eligibility_raw" must belong to that specific program (תנאי קבלה section).
- Remove page markers and decorations.
- Return **pure JSON only**, UTF-8, no markdown fences.

TEXT:
{full_text}
"""

_FILTER_PATTERN = re.compile(
    r"^(אם |במסגרת |מעולים|ההזדמנות|מקצוע |באקדמית|בתעשייה|שואפים|בעל תואר|ובני נוער|התוכנית|תוכנית|לתואר מוסמך)",
    re.UNICODE
)


def _strip(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    s = re.sub(r",\s*]", "]", s)
    s = re.sub(r",\s*}", "}", s)
    return s.strip()


def extract_majors_with_gemini(full_text: str) -> List[Dict[str, Any]]:
    load_dotenv()
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("LangChain Google GenAI not installed")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.1,
        max_output_tokens=8192,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _PROMPT)
    ])
    chain = prompt | model
    resp = chain.invoke({"full_text": full_text})
    raw = getattr(resp, "content", None) or str(resp)
    raw = _strip(raw)

    # Try parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"(\[.*\])", raw, flags=re.S)
        if m:
            data = json.loads(m.group(1))
        else:
            data = []
    if not isinstance(data, list):
        return []

    # Filter only valid degree titles
    clean: List[Dict[str, Any]] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        name = str(d.get("original_name", "")).strip()
        if not name or len(name) < 6:
            continue
        if _FILTER_PATTERN.search(name):
            continue
        if not re.search(r"(תואר|B\.A|B\.Sc|M\.A|M\.Sc|BSN|MBA)", name):
            continue

        d.setdefault("english_name", name)
        d.setdefault("keywords", [])
        d.setdefault("sample_courses", [])
        d.setdefault("eligibility_raw", "")
        d.setdefault("eligibility_rules", {})
        d["source"] = "majors.pdf"
        clean.append(d)

    return clean
