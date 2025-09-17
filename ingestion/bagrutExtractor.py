# ingestion/bagrut_extractor.py
from __future__ import annotations
import io, os, json, re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# PDF text (already in your repo)
try:
    from .PDFProcessor import PDFProcessor
except Exception:
    from PDFProcessor import PDFProcessor  # if PDFProcessor.py is at repo root

# Gemini via LangChain (same stack you use elsewhere)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------- Data model ---------------------------

@dataclass
class BagrutItem:
    subject: str
    units: float
    grade: float
    year: Optional[int] = None
    term: Optional[str] = None


# --------------------------- Utils ---------------------------

_CODEBLOCK_START = re.compile(r"^\s*```(?:json)?", re.IGNORECASE)
_CODEBLOCK_END   = re.compile(r"\s*```\s*$")

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = _CODEBLOCK_START.sub("", s)
    s = _CODEBLOCK_END.sub("", s)
    return s.strip()

def _safe_json_loads(s: str, default: Any = None) -> Any:
    """Robust JSON parse with light repairs (like in your other modules)."""
    if not s:
        return default
    s1 = _strip_code_fences(s)
    try:
        return json.loads(s1)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s1, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    s2 = re.sub(r",\s*([}\]])", r"\1", s1)  # trailing commas
    s2 = s2.replace("'", '"')
    try:
        return json.loads(s2)
    except Exception:
        return default


# --------------------------- Prompt ---------------------------

_SYSTEM = (
    "You are a precise data extractor. Return ONLY valid JSON. "
    "Extract Bagrut (Israeli matriculation) subjects, units (Yechidot Limud), and grades. "
    "Be tolerant to Hebrew/English text and messy layouts."
)

_USER_TMPL = (
    "From the following PDF text, extract an array named 'items' where each item is:\n"
    "{{\n"
    "  \"subject\": string,          // e.g., \"מתמטיקה\", \"English\", \"מדעי המחשב\"\n"
    "  \"units\": number,            // e.g., 3, 4, 5 (float allowed)\n"
    "  \"grade\": number,            // 0..100 (float allowed)\n"
    "  \"year\": number|null,        // 4-digit year if present, else null\n"
    "  \"term\": string|null         // e.g., \"מועד קיץ\", \"Winter\", if present\n"
    "}}\n\n"
    "Rules:\n"
    "- If multiple attempts for the same subject appear, keep the BEST grade (highest) with its units.\n"
    "- Ignore preparatory/non-grade lines (e.g., headers/footers).\n"
    "- JSON only; no commentary; no code fences.\n\n"
    "TEXT:\n---\n{doc_text}\n---\n"
    "Return JSON object with a single key 'items': [ ... ]"
)


# --------------------------- Core extractor ---------------------------

class BagrutExtractor:
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0):
        """
        Requires GOOGLE_API_KEY in environment or .env (same as your other modules).
        Default model follows GEMINI_MODEL or 'gemini-2.0-flash'.
        """
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Put it in your .env")
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human",  _USER_TMPL),
        ])

    def _extract_text(self, pdf_path: str) -> str:
        with open(pdf_path, "rb") as f:
            data = io.BytesIO(f.read())
        return PDFProcessor.extract_text_from_pdf(data) or ""

    def extract(self, pdf_path: str) -> List[BagrutItem]:
        """
        1) PDF -> text (PDFProcessor)
        2) Gemini -> JSON { items: [...] }
        3) Coerce, dedupe (keep best grade per subject), return list[BagrutItem]
        """
        text = self._extract_text(pdf_path)
        raw = (self.prompt | self.llm | self.parser).invoke({"doc_text": text})
        data = _safe_json_loads(raw, default={}) or {}
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            items = []

        parsed: List[BagrutItem] = []
        for obj in items:
            try:
                subj = str(obj.get("subject", "")).strip()
                if not subj:
                    continue
                units = float(obj.get("units", 0))
                grade = float(obj.get("grade", -1))
                year  = int(obj["year"]) if obj.get("year") not in (None, "", "null") else None
                term  = str(obj["term"]).strip() if obj.get("term") not in (None, "", "null") else None
            except Exception:
                # coercion fallback
                subj = str(obj.get("subject", "")).strip()
                if not subj:
                    continue
                try:
                    units = float(re.sub(r"[^\d.]", "", str(obj.get("units", "0")) or "0"))
                except Exception:
                    units = 0.0
                try:
                    grade = float(re.sub(r"[^\d.]", "", str(obj.get("grade", "-1")) or "-1"))
                except Exception:
                    grade = -1.0
                year = None
                term = None

            if units <= 0 or not (0 <= grade <= 100):
                continue
            parsed.append(BagrutItem(subject=subj, units=units, grade=grade, year=year, term=term))

        # dedupe: keep best grade per subject
        best: Dict[str, BagrutItem] = {}
        for it in parsed:
            key = it.subject.strip().lower()
            if key not in best or it.grade > best[key].grade:
                best[key] = it
        return list(best.values())

    @staticmethod
    def summarize(items: List[BagrutItem]) -> Dict[str, Any]:
        total_units = sum(i.units for i in items)
        weighted_avg = round(sum(i.units * i.grade for i in items) / total_units, 2) if total_units > 0 else None
        by_subject = {
            it.subject: {"best_grade": it.grade, "units": it.units, "year": it.year, "term": it.term}
            for it in items
        }
        return {
            "total_units": total_units,
            "weighted_average": weighted_avg,
            "by_subject": by_subject,
        }

    def run(self, pdf_path: str, out_json: Optional[str] = None) -> Tuple[List[BagrutItem], Dict[str, Any]]:
        """Convenience: extract + summarize (+ optional save)."""
        items = self.extract(pdf_path)
        summary = self.summarize(items)
        if out_json:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            payload = {
                "items": [asdict(i) for i in items],
                "summary": summary,
                "source": os.path.basename(pdf_path),
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return items, summary
