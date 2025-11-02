"""
MajorsExtractor — programs + courses + eligibility (Gemini 2 Flash + fallback)
-------------------------------------------------------------------------------
• Reads text from PDFProcessor (fallback: pdfplumber)
• Sends full extracted text once to Gemini 2 Flash via generation/majors_llm_extract.py
• Returns JSON array of majors with eligibility rules
• If LLM yields 0 items → fallback heuristic parser
"""
from __future__ import annotations
from generation.majors_llm_extract import extract_majors_with_gemini
import os, re, io, json, unicodedata as ud
from typing import List, Dict, Any, Optional
from functools import lru_cache
from dotenv import load_dotenv

# ---- PDF text ----
try:
    from .pdf_processor import PDFProcessor  # type: ignore
except Exception:
    PDFProcessor = None  # type: ignore

# ---- translator ----
try:
    from academic_advisor_backend.translator import MajorsTranslator
except Exception:
    MajorsTranslator = None
try:
    import academic_advisor_backend.translator as _translator_mod
except Exception:
    _translator_mod = None

# ---- Gemini Flash Extractor ----
try:
    from generation.majors_llm_extract import extract_majors_with_gemini
except Exception:
    extract_majors_with_gemini = None

DEBUG = os.getenv("MAJORS_DEBUG", "0") == "1"


def _dbg(*a: Any):
    if DEBUG:
        print("[majors_extractor:debug]", *a, flush=True)


@lru_cache(maxsize=4096)
def _has_hebrew(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (0x0590 <= cp <= 0x05FF) or (0xFB1D <= cp <= 0xFB4F):
            if ud.category(ch).startswith("L"):
                return True
    return False


# ========================= Text Extraction =========================
def _extract_text(pdf_path: str) -> tuple[str, Dict[str, Any]]:
    markers_proc = markers_plumber = 0
    text_proc = ""
    if PDFProcessor is not None:
        try:
            with open(pdf_path, "rb") as f:
                text_proc = PDFProcessor.extract_text_from_pdf(io.BytesIO(f.read())) or ""
            markers_proc = len(re.findall(r"--- Page \d+ ---", text_proc))
        except Exception as e:
            _dbg("PDFProcessor failed:", e)

    if text_proc.strip():
        meta = {"chosen": "PDFProcessor", "markers_processor": markers_proc, "markers_plumber": 0}
        return text_proc, meta

    try:
        import pdfplumber
    except Exception:
        return "", {"chosen": "none"}
    try:
        chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                t = page.extract_text() or ""
                chunks.append(f"--- Page {i} ---\n{t}\n")
        text_plumber = "\n".join(chunks)
        markers_plumber = len(re.findall(r"--- Page \d+ ---", text_plumber))
        return text_plumber, {"chosen": "pdfplumber", "markers_plumber": markers_plumber}
    except Exception as e:
        _dbg("pdfplumber failed:", e)
        return "", {"chosen": "error"}


def _write_debug_text_dump(out_json: str, text: str, meta: dict):
    try:
        base_dir = os.path.dirname(os.path.abspath(out_json)) or os.getcwd()
        dump_path = os.path.join(base_dir, "majors_text_debug.txt")
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write(text or "")
        print(f"[majors_extractor] Debug text dump -> {dump_path}")
        if meta:
            print(f"[majors_extractor] Text source: {meta.get('chosen')}")
    except Exception as e:
        print(f"[majors_extractor:warn] failed writing debug dump: {e}")


# ========================= Simple Rules Merge =========================
def _pick_max_int(a, b):
    if a is None: return b
    if b is None: return a
    return max(int(a), int(b))


def _merge_rules(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not a: return b or {}
    if not b: return a or {}
    out = dict(a)
    for k, v in b.items():
        if k == "subjects":
            out.setdefault("subjects", {})
            for sname, sreq in (v or {}).items():
                cur = out["subjects"].get(sname, {})
                merged = {
                    "min_units": _pick_max_int(cur.get("min_units"), sreq.get("min_units")),
                    "min_grade": _pick_max_int(cur.get("min_grade"), sreq.get("min_grade")),
                }
                out["subjects"][sname] = merged
        elif k in ("psychometric_min", "bagrut_avg_min"):
            out[k] = _pick_max_int(out.get(k), v)
        elif k == "english_level_min":
            out[k] = out.get(k) or v
        elif k == "interview_required":
            out[k] = bool(out.get(k) or v)
        elif k == "notes":
            out.setdefault("notes", [])
            out["notes"] = list(dict.fromkeys((out["notes"] or []) + (v or [])))
        else:
            out[k] = out.get(k) or v
    return out


def _normalize_key(name: str) -> str:
    s = re.sub(r"\s+", " ", name or "").strip()
    return re.sub(r"[:：]+$", "", s)


def _is_program_name(name: str) -> bool:
    tokens = ["תואר", "B.A", "B.Sc", "BSN", "M.A", "M.Sc", "הסבת אקדמאים"]
    return any(t in name for t in tokens)


def _merge_program_rows(rows: List[Dict[str, Any]], source_path: str) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = _normalize_key(r.get("original_name", ""))
        if not key:
            continue
        cur = merged.get(key)
        if not cur:
            merged[key] = dict(r)
        else:
            cur["sample_courses"] = list(dict.fromkeys(cur.get("sample_courses", []) + r.get("sample_courses", [])))
            cur["keywords"] = list(dict.fromkeys(cur.get("keywords", []) + r.get("keywords", [])))
            if len(r.get("eligibility_raw", "")) > len(cur.get("eligibility_raw", "")):
                cur["eligibility_raw"] = r["eligibility_raw"]
            cur["eligibility_rules"] = _merge_rules(cur.get("eligibility_rules", {}), r.get("eligibility_rules", {}))
    return list(merged.values())


# ========================= Fallback Parser =========================
def _fallback_parse_items(text: str) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    items = []
    title = ""
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if _is_program_name(s):
            title = _normalize_key(s)
            i += 1
            courses, elig_lines = [], []
            captured_elig = False
            for j in range(i, min(len(lines), i + 80)):
                line = lines[j].strip()
                if _is_program_name(line):
                    break
                if any(h in line for h in ["תנאי קבלה", "דרישות קבלה", "סף קבלה"]):
                    captured_elig = True
                if captured_elig:
                    elig_lines.append(line)
                if any(x in line for x in ["מבוא", "יסודות", "קורס", "ניהול", "סטטיסט", "אלגוריתם", "כלכלה", "קריאה", "כתיבה"]):
                    courses.append(line)
            items.append({
                "original_name": title,
                "sample_courses": list(dict.fromkeys(courses))[:20],
                "keywords": [],
                "eligibility_raw": "\n".join(elig_lines).strip(),
                "eligibility_rules": {},
            })
        i += 1
    return items


# ========================= Translator =========================
def _translate_names(names: List[str]) -> Dict[str, str]:
    if MajorsTranslator:
        try:
            return MajorsTranslator().translate_name_map(names)
        except Exception as e:
            _dbg("MajorsTranslator failed:", e)
    if _translator_mod:
        tx = getattr(_translator_mod, "translate_batch", None)
        if callable(tx):
            try:
                res = tx(names, target_lang="en")
                return {names[i]: res[i] for i in range(len(names))}
            except Exception as e:
                _dbg("translate_batch failed:", e)
    return {n: n for n in names}


# ========================= MAIN FUNCTION =========================
def build_majors_profiles(pdf_path: str, out_json: str = "extracted_majors.json") -> str:
    load_dotenv()
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    text, meta = _extract_text(pdf_path)
    _write_debug_text_dump(out_json, text, meta)

    # Try Gemini 2 Flash extraction
    items: List[Dict[str, Any]] = []
    if extract_majors_with_gemini and os.getenv("MAJORS_NO_LLM", "0") != "1":
        try:
            items = extract_majors_with_gemini(text)
        except Exception as e:
            _dbg("Gemini extraction failed:", e)

    if not items:
        print("[majors_extractor] LLM returned 0 → fallback parser.")
        items = _fallback_parse_items(text)

    # Normalize + merge
    prelim = []
    for it in items:
        name = str(it.get("original_name", "")).strip()
        if not name:
            continue
        prelim.append({
            "original_name": name,
            "english_name": "",
            "keywords": it.get("keywords", []),
            "sample_courses": it.get("sample_courses", []),
            "eligibility_raw": it.get("eligibility_raw", ""),
            "eligibility_rules": it.get("eligibility_rules", {}),
            "source": os.path.basename(pdf_path),
        })

    merged = _merge_program_rows(prelim, os.path.basename(pdf_path))

    # Translate names
    names = [r["original_name"] for r in merged]
    name_map = _translate_names(names)
    for r in merged:
        en = name_map.get(r["original_name"], r["original_name"])
        if _has_hebrew(en):
            en = r["original_name"]
        r["english_name"] = en

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[majors_extractor] Wrote {len(merged)} majors -> {out_json}")
    return out_json
