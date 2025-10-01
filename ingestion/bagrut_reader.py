# ingestion/bagrut_reader.py
from __future__ import annotations
import io, os, json, re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# PDF text (your repo ships PDFProcessor.py)
try:
    from .PDFProcessor import PDFProcessor
except Exception:
    try:
        from .pdf_processor import PDFProcessor
    except Exception:
        from PDFProcessor import PDFProcessor  # final fallback

# Try to import Gemini via LangChain (optional fallback)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    StrOutputParser = None


# --------------------------- Data model ---------------------------

@dataclass
class SubjectAverage:
    subject: str
    final_grade: float
    units: Optional[float] = None
    year: Optional[int] = None


# --------------------------- Utils ---------------------------

def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u200f", "").replace("\u200e", "")
    return re.sub(r"\s+", " ", s).strip()

def _normalize_subject_name(name: str) -> str:
    s = _norm(name)
    s = re.sub(r'\s*יח"?ל.*$', "", s).strip()                        # strip unit annotations (יח"ל ...)
    s = re.sub(r"\s+שעות\s+\d+(\.\d+)?$", "", s).strip()            # strip hour rows
    s = s.replace(' בי"ס', "").replace(' (בי"ס ערבי', "").strip()   # drop school markers
    s = s.split(":")[0].strip()                                     # 'עברית: הבעה...' -> 'עברית'
    parts = [p.strip() for p in re.split(r"\s*[-–]\s*", s) if p.strip()]
    if parts:
        s = parts[0]
    return s

def _safe_json_loads(s: str, default: Any = None) -> Any:
    if not s:
        return default
    s1 = s.strip()
    if s1.startswith("```"):
        s1 = re.sub(r"^\s*```(?:json)?", "", s1, flags=re.I).strip()
        s1 = re.sub(r"```$", "", s1).strip()
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
    s2 = re.sub(r",\s*([}\]])", r"\1", s1).replace("'", '"')
    try:
        return json.loads(s2)
    except Exception:
        return default


# prefer real 3-digit finals (050–099) or 100; fallback to clean 2-digit (50–99) if OCR dropped the zero
THREE_DIGIT_GRADE_RX = re.compile(r'^(?:0(?:5\d|6\d|7\d|8\d|9\d)|100)$')  # 050–099 or 100
TWO_DIGIT_GRADE_RX   = re.compile(r'^(?:5\d|6\d|7\d|8\d|9\d)$')           # 50–99

UNITS_RX  = re.compile(r'(?:(\d(?:\.\d)?)\s*יח)|(?:(\d)\s*/\s*(\d)\s*יח)')
DATE_RX   = re.compile(r"\b(?P<m>0[1-9]|1[0-2])/(?P<y>20\d{2})\b")
ADJ_RX    = re.compile(r"(מצוין|טוב מאוד|טוב|נכשל)")

def _parse_units(text: str) -> Optional[float]:
    m = UNITS_RX.search(text or "")
    if not m: return None
    if m.group(1):
        try: return float(m.group(1))
        except Exception: return None
    try:
        a = float(m.group(2)); b = float(m.group(3))
        return max(a, b)
    except Exception:
        return None

def _parse_grade(text: str) -> Tuple[Optional[int], str]:
    """
    Returns (grade_or_none, reason). Prefers 3-digit finals (050–099, 100); then 2-digit (50–99).
    Rejects codes like 001/003/006/000.
    """
    c = _norm(text)
    tokens = re.findall(r"\d{2,3}", c)
    if not tokens:
        return None, "no 2–3 digit token"
    # check from rightmost (final numbers usually at end)
    for t in reversed(tokens):
        if len(t) == 3 and THREE_DIGIT_GRADE_RX.match(t):
            return int(t), "3-digit"
    for t in reversed(tokens):
        if len(t) == 2 and TWO_DIGIT_GRADE_RX.match(t):
            return int(t), "2-digit"
    return None, "looks like a code (e.g., 001/003/006/000)"

def _parse_date(line: str) -> Tuple[Optional[int], Optional[int]]:
    m = DATE_RX.search(line or "")
    if not m: return None, None
    try: return int(m.group("y")), int(m.group("m"))
    except Exception: return None, None

# --- OCR fix for 90x→00x on starred lines (e.g., 093→003, 098→008, 090→000) ---
_OCR_90X_AT_END = re.compile(r'(^.*\D)00([0-9])(\s*\*?\s*)$')
def _fix_ocr_90x_bug(line: str) -> Tuple[str, bool]:
    m = _OCR_90X_AT_END.match(line)
    if not m:
        return line, False
    fixed = m.group(1) + "09" + m.group(2) + m.group(3)
    return fixed, (fixed != line)


# --------------------------- LLM prompt (optional fallback) ---------------------------

_SYSTEM = (
    "You extract FINAL per-subject results from Israeli Bagrut. "
    "Return ONLY valid JSON."
)

# Only {doc} is a real variable; braces are escaped.
_USER_TMPL = (
    "The following lines are ONLY the rows that contain an asterisk '*' in a Bagrut transcript "
    "(i.e., finals/summary rows). Some OCR errors replace 09x with 00x at the end; treat 00d at the end as 09d.\n\n"
    "TEXT:\n---\n{doc}\n---\n\n"
    "Task: For each subject, extract ONE final record with:\n"
    "- subject (string)\n"
    "- final_grade (number 50..100). Prefer a 3-digit final like 078/093/098/090 or 100; alternatively 2-digit 50..99.\n"
    "- units (number or null)\n"
    "- year (4-digit year or null)\n\n"
    "Rules:\n"
    "1) Consider only lines that include '*'.\n"
    "2) If the ending number looks like 00d (e.g., 003/006/008/000/001), interpret it as 09d (093/096/098/090/091).\n"
    "3) Ignore rows that still don't contain a valid 50..100 final number after the fix.\n"
    "4) If the subject appears multiple times, keep the latest year (and the highest grade if years tie).\n"
    "5) Omit anything with the word 'שעות'.\n\n"
    "Return JSON only:\n"
    "{{\n"
    "  \"subjects\": [\n"
    "    {{\"subject\": string, \"final_grade\": number, \"units\": number|null, \"year\": number|null}}\n"
    "  ]\n"
    "}}\n"
)

# --------------------------- Reader ---------------------------

class BagrutReader:
    """
    Deterministic star-only extractor with OCR 90x→00x fix, plus optional Gemini fallback.

    Args:
        debug (bool): print per-line starred summary and dump full text to debug_bagrut_text.txt.
        debug_dump_path (str): where to write the extracted text dump.
        strict_star_only (bool): kept for compatibility (True keeps star-only path; there is no non-star fallback here).
        use_llm_fallback (bool): if True and some finals are still missed, run Gemini on the *starred lines only* and merge.
        model_name (str|None): Gemini model name if fallback is enabled.
        temperature (float): Gemini temperature (0.0 recommended).
    """

    def __init__(
        self,
        debug: bool = True,
        debug_dump_path: str = "debug_bagrut_text.txt",
        strict_star_only: bool = True,
        use_llm_fallback: bool = False,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        load_dotenv()
        self.debug = debug
        self.debug_dump_path = debug_dump_path
        self.strict_star_only = strict_star_only  # kept for app.py arg compatibility
        self.use_llm_fallback = use_llm_fallback

        # Optional LLM
        self.llm = None
        self.parser = None
        self.prompt = None
        if self.use_llm_fallback and ChatGoogleGenerativeAI and ChatPromptTemplate and StrOutputParser:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                try:
                    self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
                    self.parser = StrOutputParser()
                    self.prompt = ChatPromptTemplate.from_messages([
                        ("system", _SYSTEM),
                        ("human",  _USER_TMPL),
                    ])
                except Exception as e:
                    print(f"[bagrut:warn] LLM fallback disabled (init failed): {e}")
                    self.llm = None
            else:
                print("[bagrut:warn] LLM fallback disabled (missing GOOGLE_API_KEY).")

    @staticmethod
    def _extract_text(pdf_path: str) -> str:
        with open(pdf_path, "rb") as f:
            data = io.BytesIO(f.read())
        return PDFProcessor.extract_text_from_pdf(data) or ""

    def _debug_dump(self, text: str) -> None:
        # write full text
        try:
            with open(self.debug_dump_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

        lines = text.splitlines()
        star_lines = [ln for ln in lines if "*" in ln]
        page_count = text.count("\n--- Page ") or 1

        print(f"[bagrut:debug] pages≈{page_count}, lines={len(lines)}, starred_lines={len(star_lines)}")
        shown = 0
        for ln in star_lines[:25]:
            if "שעות" in ln:
                print("  • [ignored: שעות] ", _norm(ln)); shown += 1; continue
            ln_fixed, changed = _fix_ocr_90x_bug(ln)
            g, reason = _parse_grade(ln_fixed)
            tag = " (OCR-fixed)" if changed else ""
            if g is None:
                print(f"  • [ignored: {reason}{tag}] ", _norm(ln_fixed))
            else:
                print(f"  • [accepted {g} ({reason}){tag}] ", _norm(ln_fixed))
            shown += 1
        if len(star_lines) > shown:
            print(f"  … and {len(star_lines) - shown} more starred lines")
        print(f"[bagrut:debug] full extracted text saved to: {os.path.abspath(self.debug_dump_path)}")

    def _extract_from_starred_text(self, text: str) -> List[SubjectAverage]:
        buckets: Dict[str, Dict[str, Any]] = {}

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or "*" not in line or "שעות" in line:
                continue

            # OCR fix first
            line_fixed, _changed = _fix_ocr_90x_bug(line)

            g, _ = _parse_grade(line_fixed)
            if g is None or not (50 <= g <= 100):
                continue

            nostar = line_fixed.replace("*", "").strip()

            # subject isolation: 'MM/YYYY <code> <SUBJECT ...> [יח"ל ...] <ADJ?> <GRADE>'
            m_code = re.search(r"\b\d{2}/\d{4}\s+\d+\s+(?P<rest>.+)$", nostar)
            if m_code:
                rest = m_code.group("rest")
                m_adj = ADJ_RX.search(rest)
                subj = rest[:m_adj.start()] if m_adj else re.split(r"\s+\d{2,3}\s*$", rest)[0]
            else:
                parts = re.split(r"\s+\d{2,3}\s*$", nostar)
                subj = parts[0] if parts else ""

            subj = _normalize_subject_name(subj)
            if not subj:
                continue

            units = _parse_units(nostar)
            y, m = _parse_date(nostar)
            order = (y or 0, m or 0)

            key = subj.lower()
            cur = buckets.get(key)
            if (not cur) or (order > cur["order"]) or (g > cur["final_grade"] and order == cur["order"]):
                buckets[key] = {
                    "subject": subj,
                    "final_grade": float(g),
                    "units": units,
                    "year": y,
                    "order": order,
                }

        return [
            SubjectAverage(
                subject=v["subject"],
                final_grade=round(v["final_grade"], 2),
                units=v.get("units"),
                year=v.get("year"),
            )
            for v in buckets.values()
        ]

    def _llm_fill_from_starred(self, full_text: str, have: List[SubjectAverage]) -> List[SubjectAverage]:
        """
        Run Gemini ONLY on starred lines to fill gaps.
        Deterministic 'have' stays authoritative; LLM adds missing subjects.
        """
        if not (self.llm and self.prompt and self.parser):
            return have

        # keep only starred lines (what the prompt expects)
        starred_lines = [ln for ln in full_text.splitlines() if "*" in ln and "שעות" not in ln]
        if not starred_lines:
            return have

        raw = (self.prompt | self.llm | self.parser).invoke({"doc": "\n".join(starred_lines)})
        data = _safe_json_loads(raw, default={}) or {}
        arr = data.get("subjects") if isinstance(data, dict) else None
        if not isinstance(arr, list):
            return have

        # normalize & merge
        have_by_key = {s.subject.strip().lower(): s for s in have}
        for x in arr:
            subj = _normalize_subject_name(x.get("subject", ""))
            if not subj:
                continue
            # try the same OCR correction logic on LLM-provided subject lines is not needed here;
            # we expect LLM to output numeric final_grade already.
            try:
                g = float(x.get("final_grade", x.get("grade", -1)))
            except Exception:
                continue
            if not (50 <= g <= 100):
                continue
            units = None
            u = x.get("units")
            try:
                if u not in (None, "", "null"):
                    units = float(u)
                    if units < 0: units = None
            except Exception:
                units = None
            year = None
            yr = x.get("year")
            if isinstance(yr, (int, float)) or (isinstance(yr, str) and yr.isdigit()):
                year = int(yr)

            key = subj.lower()
            if key in have_by_key:
                # keep deterministic; upgrade year/units if missing
                cur = have_by_key[key]
                if cur.year is None and year is not None:
                    cur.year = year
                if cur.units is None and units is not None:
                    cur.units = units
            else:
                have_by_key[key] = SubjectAverage(subject=subj, final_grade=round(g, 2), units=units, year=year)

        return list(have_by_key.values())

    # ---------- Public API ----------

    def run(self, pdf_path: str, out_json: Optional[str] = None) -> Tuple[List[SubjectAverage], Dict[str, Any]]:
        text = self._extract_text(pdf_path)
        if self.debug:
            self._debug_dump(text)

        # 1) deterministic star-only with OCR fix
        subjects = self._extract_from_starred_text(text)

        # 2) optional Gemini fallback (starred-only text)
        if self.use_llm_fallback and (not subjects or len(subjects) < 6):
            subjects = self._llm_fill_from_starred(text, subjects)

        print(f"[bagrut] starred finals kept: {len(subjects)}" + (" (+ LLM fallback)" if self.use_llm_fallback else ""))

        payload = {"subjects": [asdict(s) for s in subjects], "source": os.path.basename(pdf_path)}
        if out_json:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        return subjects, {"count_subjects": len(subjects)}
