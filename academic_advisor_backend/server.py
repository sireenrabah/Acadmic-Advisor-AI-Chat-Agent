# server.py
from __future__ import annotations

import os
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Optional LLM (Gemini via LangChain) ------------------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# ---- Project imports --------------------------------------------------------
# HybridRAG = session brain (Bagrut-driven questioning + recs)
from query.query import HybridRAG

# Load/normalize Bagrut JSON and compute signals
from query.bagrut_features import load_bagrut, normalize_bagrut

# Bagrut extractor (try both package styles)
try:
    from ingestion.bagrut_reader import BagrutReader
except Exception:
    try:
        from bagrut_reader import BagrutReader  # fallback if file is at project root
    except Exception:
        BagrutReader = None  # we'll gracefully handle if not available

# Psychometry extractor
try:
    from ingestion.psychometry_reader import PsychometryReader
except Exception:
    PsychometryReader = None


# =============================================================================
# Paths / constants
# =============================================================================
# put this at the top, after imports but BEFORE `from query.query import HybridRAG`
os.environ["BAGRUT_JSON_PATH"] = str((Path(__file__).parent / "state" / "extracted_bagrut.json").resolve())
BASE_DIR = Path(__file__).parent.resolve()
STATE_DIR = BASE_DIR / "state"
MAJORS_DIR = BASE_DIR / "majors_data"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MAJORS_DIR.mkdir(parents=True, exist_ok=True)
BAGRUT_UPLOADED_PATH = STATE_DIR / "uploaded_bagrut.pdf"
BAGRUT_JSON_PATH = STATE_DIR / "extracted_bagrut.json"
PSYCHOMETRY_UPLOADED_PATH = STATE_DIR / "uploaded_psychometry.pdf"
PSYCHOMETRY_JSON_PATH = STATE_DIR / "extracted_psychometry.json"
MAJORS_META_JSON = MAJORS_DIR / "extracted_majors.json"
MAJORS_EMB_JSON = MAJORS_DIR / "majors_embeddings.json"

# Make the JSON path visible to modules that read from env
os.environ["BAGRUT_JSON_PATH"] = str(BAGRUT_JSON_PATH.resolve())

# =============================================================================
# Load Majors Data (CRITICAL - needed for recommendations)
# =============================================================================
MAJORS_DATA_DIR = Path("majors_data")
EXTRACTED_MAJORS_PATH = MAJORS_DATA_DIR / "extracted_majors.json"
MAJORS_EMBEDDINGS_PATH = MAJORS_DATA_DIR / "majors_embeddings.json"

def _load_majors_data() -> List[Dict[str, Any]]:
    """Load majors with embeddings from disk."""
    majors = []
    
    # Load extracted majors (has keywords, courses, eligibility)
    if EXTRACTED_MAJORS_PATH.exists():
        with open(EXTRACTED_MAJORS_PATH, "r", encoding="utf-8") as f:
            majors = json.load(f)
    
    # Load embeddings (has vector scores per criterion)
    if MAJORS_EMBEDDINGS_PATH.exists():
        with open(MAJORS_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            embeddings_data = json.load(f)
            
            # Merge embeddings into majors by english_name
            embeddings_by_name = {}
            if isinstance(embeddings_data, list):
                for emb in embeddings_data:
                    en = emb.get("english_name", "").strip()
                    if en:
                        embeddings_by_name[en] = emb
            
            for major in majors:
                en = major.get("english_name", "").strip()
                if en in embeddings_by_name:
                    # Add vector and scores from embeddings
                    major["vector"] = embeddings_by_name[en].get("vector", [])
                    major["scores"] = embeddings_by_name[en].get("scores", {})
    
    return majors

# Load majors at startup
_MAJORS_CACHE = _load_majors_data()
print(f"[startup] Loaded {len(_MAJORS_CACHE)} majors with embeddings")

# Make the JSON path visible to modules that read from env
# =============================================================================
# App / CORS
# =============================================================================
app = FastAPI(title="Academic Advisor Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # adjust if you want to lock to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Session store
# =============================================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# LLM setup (optional)
# =============================================================================
def _maybe_make_llm() -> Optional["ChatGoogleGenerativeAI"]:
    if ChatGoogleGenerativeAI is None:
        return None
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    try:
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)
    except Exception:
        return None

_LLM = _maybe_make_llm()

# =============================================================================
# Helpers
# =============================================================================
def _read_bagrut_now() -> Dict[str, Any]:
    """Load the freshest extracted Bagrut JSON if it exists."""
    try:
        if BAGRUT_JSON_PATH.exists():
            return load_bagrut(str(BAGRUT_JSON_PATH))
    except Exception:
        pass
    return {}

def _get_session(session_id: str) -> Dict[str, Any]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s

def _seed_rag_with_disk_bagrut(rag: HybridRAG, lang: str):
    """Force-refresh session RAG with the latest extracted_bagrut.json."""
    rag.set_session_context(ui_language=lang, bagrut_json=_read_bagrut_now())

# =============================================================================
# Schemas
# =============================================================================
class StartRequest(BaseModel):
    language: str = Field("en", description="UI language code, e.g., en | he | ar")
    degree: str = Field("bachelor", description="'bachelor' | 'master' | 'both'")
    name: Optional[str] = None
    college: Optional[str] = None

class StartResponse(BaseModel):
    session_id: str
    message: str

class TurnAnswerRequest(BaseModel):
    session_id: str
    user_text: str
    last_question: str

class TurnNextRequest(BaseModel):
    session_id: str
    history: Optional[List[List[str]]] = None  # [['user','hi'],['ai','QUESTION: ...']]
    asked_questions: Optional[List[str]] = None
    hint: Optional[str] = ""

class TextResponse(BaseModel):
    text: str

# =============================================================================
# Routes
# =============================================================================
@app.get("/health")
def health():
    exists = BAGRUT_JSON_PATH.exists()
    size = BAGRUT_JSON_PATH.stat().st_size if exists else 0
    return {
        "ok": True,
        "bagrut_json_exists": exists,
        "bagrut_json_size": size,
        "sessions": len(SESSIONS),
    }

@app.get("/test-llm")
def test_llm():
    """
    Diagnostic endpoint to test if LLM is working and responding in correct languages.
    Returns detailed diagnostics about API key, connection, and language support.
    """
    results = {
        "api_key_configured": False,
        "llm_initialized": False,
        "english_test": {"success": False, "response": None},
        "hebrew_test": {"success": False, "response": None, "has_hebrew_chars": False},
        "arabic_test": {"success": False, "response": None, "has_arabic_chars": False},
        "errors": []
    }
    
    # Check 1: API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        results["api_key_configured"] = True
        results["api_key_preview"] = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    else:
        results["errors"].append("GOOGLE_API_KEY not found in environment")
        return results
    
    # Check 2: LLM initialization
    if _LLM is not None:
        results["llm_initialized"] = True
    else:
        results["errors"].append("LLM is None - initialization failed")
        return results
    
    # Check 3: English test
    try:
        response = _LLM.invoke("Say 'test' in 1 word")
        results["english_test"]["success"] = True
        results["english_test"]["response"] = response.content.strip()
    except Exception as e:
        results["english_test"]["error"] = str(e)
        results["errors"].append(f"English test failed: {e}")
    
    # Check 4: Hebrew test
    try:
        hebrew_prompt = """LANGUAGE: HEBREW
You must respond ONLY in Hebrew language.
Example: "שלום"
Write "hello" in Hebrew (1 word):"""
        response = _LLM.invoke(hebrew_prompt)
        content = response.content.strip()
        results["hebrew_test"]["success"] = True
        results["hebrew_test"]["response"] = content
        results["hebrew_test"]["has_hebrew_chars"] = any('\u0590' <= char <= '\u05FF' for char in content)
        
        if not results["hebrew_test"]["has_hebrew_chars"]:
            results["errors"].append("LLM responded to Hebrew prompt but NOT in Hebrew")
    except Exception as e:
        results["hebrew_test"]["error"] = str(e)
        results["errors"].append(f"Hebrew test failed: {e}")
    
    # Check 5: Arabic test
    try:
        arabic_prompt = """LANGUAGE: ARABIC
You must respond ONLY in Arabic language.
Example: "مرحباً"
Write "hello" in Arabic (1 word):"""
        response = _LLM.invoke(arabic_prompt)
        content = response.content.strip()
        results["arabic_test"]["success"] = True
        results["arabic_test"]["response"] = content
        results["arabic_test"]["has_arabic_chars"] = any('\u0600' <= char <= '\u06FF' for char in content)
        
        if not results["arabic_test"]["has_arabic_chars"]:
            results["errors"].append("LLM responded to Arabic prompt but NOT in Arabic")
    except Exception as e:
        results["arabic_test"]["error"] = str(e)
        results["errors"].append(f"Arabic test failed: {e}")
    
    # Summary
    results["summary"] = {
        "all_tests_passed": len(results["errors"]) == 0,
        "connection_ok": results["api_key_configured"] and results["llm_initialized"],
        "language_enforcement_ok": (
            results["hebrew_test"].get("has_hebrew_chars", False) and 
            results["arabic_test"].get("has_arabic_chars", False)
        )
    }
    
    return results

@app.post("/start", response_model=StartResponse)
def start(req: StartRequest) -> StartResponse:
    """
    Creates a session and returns a structured 3-part message:
    1. College greeting (if provided)
    2. Simple welcome
    3. Bagrut acknowledgment with grades
    4. First question
    """
    sid = str(uuid.uuid4())
    
    # DEBUG: Log what language was received from frontend
    print(f"[start] ═══════════════════════════════════════════")
    print(f"[start] NEW SESSION: {sid[:8]}")
    print(f"[start] Language requested: '{req.language}'")
    print(f"[start] Degree: {req.degree}")
    print(f"[start] Name: {req.name}")
    print(f"[start] ═══════════════════════════════════════════")
    
    rag = HybridRAG(ui_language=req.language, llm=_LLM)
    
    # Set degree filter from user's choice
    rag.degree_filter = req.degree if req.degree in ["bachelor", "master", "both"] else "both"
    print(f"[start] Degree filter set to: {rag.degree_filter}")

    # Load majors data into RAG (CRITICAL for recommendations)
    rag.majors = _MAJORS_CACHE.copy()  # Give each session a copy
    print(f"[start] Session {sid[:8]} loaded {len(rag.majors)} majors")

    # Seed with whatever Bagrut we already have on disk (if any)
    _seed_rag_with_disk_bagrut(rag, req.language)
    
    # DEBUG: Check if Bagrut was loaded
    bagrut_subjects = len((rag.bagrut_json or {}).get("subjects", []))
    by_subject_count = len(normalize_bagrut(rag.bagrut_json or {}).get("by_subject", {}))
    print(f"[start:debug] Bagrut loaded: {bagrut_subjects} subjects raw, {by_subject_count} normalized")

    # Store session
    SESSIONS[sid] = {
        "rag": rag,
        "lang": req.language,
        "degree": req.degree,
        "name": (req.name or "").strip(),
        "college": (req.college or "").strip(),
        "turn_count": 0,  # Track turns for stopping logic
    }

    # Use new 3-part greeting method
    result = rag.greet_and_first_question()
    
    # Build structured message with clear separators
    message_parts = []
    
    # 1. College-specific greeting if provided
    college_name = (req.college or "").strip()
    if college_name:
        try:
            from translator import UITranslator
            ui_tr = UITranslator()
            college_greeting = ui_tr.tr(req.language, f"Great to see you're interested in {college_name}!")
            message_parts.append(college_greeting)
        except Exception:
            pass  # Skip college greeting if translator fails
    
    # 2. Simple welcome (SKIP if college greeting exists to avoid duplication)
    if not college_name and result.get('greeting') and result['greeting'].strip():
        message_parts.append(result['greeting'])
    
    # 3. Bagrut acknowledgment (if available)
    if result.get('bagrut_summary'):
        message_parts.append(result['bagrut_summary'])
    
    # 4. First question
    message_parts.append(result['first_question'])
    
    # Join with double newlines for clear separation
    # Note: Frontend will detect college greeting (first part) and render it BOLD in separate visual section
    message = "\n\n".join(message_parts)
    
    return StartResponse(session_id=sid, message=message)

@app.post("/upload")
async def upload(
    session_id: str = Form(...),
    bagrut: Optional[UploadFile] = File(None),
    psychometry: Optional[UploadFile] = File(None),
):
    """
    Accepts Bagrut and/or Psychometry files.
    - Bagrut: saves to /state/uploaded_bagrut.pdf, extracts to /state/extracted_bagrut.json
    - Psychometry: saves to /state/uploaded_psychometry.pdf, extracts to /state/extracted_psychometry.json
    Then reseeds the session with both.
    """
    # Save and extract Bagrut
    if bagrut is not None:
        suffix = ""
        if bagrut.filename:
            suffix = Path(bagrut.filename).suffix or ".pdf"
        target_path = BAGRUT_UPLOADED_PATH
        with open(target_path, "wb") as f:
            content = await bagrut.read()
            f.write(content)

        # Run Bagrut extractor if available
        if BagrutReader is not None:
            try:
                reader = BagrutReader(strict_star_only=True, use_llm_fallback=True, debug=True)
                items, summary = reader.run(str(target_path), out_json=str(BAGRUT_JSON_PATH))
                print(f"[bagrut] extracted {len(items)} items -> {BAGRUT_JSON_PATH.name}")
                os.environ["BAGRUT_JSON_PATH"] = str(BAGRUT_JSON_PATH.resolve())
            except Exception as e:
                print(f"[bagrut:error] {e}")
        else:
            if not BAGRUT_JSON_PATH.exists():
                BAGRUT_JSON_PATH.write_text('{"by_subject": {}, "total_units": 0}', encoding="utf-8")
            os.environ["BAGRUT_JSON_PATH"] = str(BAGRUT_JSON_PATH.resolve())
    
    # Save and extract Psychometry
    if psychometry is not None:
        suffix = ""
        if psychometry.filename:
            suffix = Path(psychometry.filename).suffix or ".pdf"
        target_path = PSYCHOMETRY_UPLOADED_PATH
        with open(target_path, "wb") as f:
            content = await psychometry.read()
            f.write(content)

        # Run Psychometry extractor if available
        if PsychometryReader is not None:
            try:
                reader = PsychometryReader(use_llm_fallback=True, debug=True)
                scores, summary = reader.run(str(target_path), out_json=str(PSYCHOMETRY_JSON_PATH))
                print(f"[psychometry] extracted scores -> {PSYCHOMETRY_JSON_PATH.name}")
                print(f"[psychometry] {summary}")
                
                # Store psychometry scores in session for easy access
                sess = SESSIONS.get(session_id)
                if sess:
                    sess["psychometry"] = scores
                    
            except Exception as e:
                print(f"[psychometry:error] {e}")
        else:
            if not PSYCHOMETRY_JSON_PATH.exists():
                PSYCHOMETRY_JSON_PATH.write_text('{"overall_score": null}', encoding="utf-8")

    # Reseed the current session with the freshly extracted data
    sess = SESSIONS.get(session_id)
    if sess:
        rag: HybridRAG = sess["rag"]
        lang = sess.get("lang") or "en"
        _seed_rag_with_disk_bagrut(rag, lang)

    return {"ok": True}

@app.post("/turn/answer")
def turn_answer(req: TurnAnswerRequest):
    """
    Blend the user's last answer into the person profile.
    """
    import sys
    print(f"[/turn/answer] ==================== RECEIVED REQUEST ====================", flush=True)
    print(f"[/turn/answer] Session: {req.session_id}", flush=True)
    print(f"[/turn/answer] User text: '{req.user_text[:80]}...'", flush=True)
    print(f"[/turn/answer] Last question: '{req.last_question[:80]}...'", flush=True)
    sys.stdout.flush()
    
    sess = _get_session(req.session_id)
    rag: HybridRAG = sess["rag"]
    
    print(f"[/turn/answer] About to call rag.absorb_answer()", flush=True)
    ok = rag.absorb_answer(user_text=req.user_text, last_question=req.last_question)
    print(f"[/turn/answer] absorb_answer returned: {ok}", flush=True)
    sys.stdout.flush()
    
    return {"ok": bool(ok)}

@app.post("/turn/next", response_model=TextResponse)
def turn_next(req: TurnNextRequest) -> TextResponse:
    """
    Produce the next interview question OR inline recommendations if ready.
    Always refresh from disk so the chat immediately uses newly uploaded Bagrut.
    
    Returns:
    - Regular question: {"text": "Question text"}
    - Recommendations ready: {"text": "__RECOMMEND__\nTransition message\n\nTop 3 majors..."}
    """
    sess = _get_session(req.session_id)
    rag: HybridRAG = sess["rag"]
    lang = sess.get("lang") or "en"

    # Increment turn count
    sess["turn_count"] = sess.get("turn_count", 0) + 1
    turn_count = sess["turn_count"]

    # Force-refresh from disk before generating the next question
    _seed_rag_with_disk_bagrut(rag, lang)
    
    # Check if we should stop and recommend (with vector convergence check)
    if rag.should_recommend_now(history=req.history or [], turn_count=turn_count):
        # Generate transition message
        transition = rag.transition_to_recommendations()
        
        # Get structured recommendations (with all fields)
        structured_recs = rag.recommend_final(top_k=3)
        
        # CRITICAL: Store structured recommendations in session for detail fetching
        sess["recommended_majors"] = structured_recs
        print(f"[turn/next] Stored {len(structured_recs)} structured recommendations in session")
        for i, rec in enumerate(structured_recs):
            print(f"  [{i}] original_name='{rec.get('original_name')}', score={rec.get('score')}")
        
        # Generate inline text for display
        inline_recs = rag.format_inline_recommendations(top_k=3)
        
        # Combine with signal prefix for frontend detection
        message = f"__RECOMMEND__\n{transition}\n\n{inline_recs}"
        
        return TextResponse(text=message)

    text = rag.ask_next_question(
        history=req.history or [],
        asked_questions=req.asked_questions or [],
        hint=req.hint or "",
    )
    return TextResponse(text=text)

@app.post("/finish")
def finish(payload: Dict[str, Any]):
    """
    End the session (frontend calls this when user exits).
    We keep it simple: just forget the session if present.
    """
    session_id = payload.get("session_id")
    if session_id and session_id in SESSIONS:
        SESSIONS.pop(session_id, None)
    return {"ok": True}

@app.post("/turn/followup", response_model=TextResponse)
def turn_followup(req: Dict[str, Any]) -> TextResponse:
    """
    Handle follow-up option clicks after recommendations.
    First asks user to select a major number (1-3).
    Then provides 4 detailed options for that major.
    
    NEW: Supports `major_index` parameter for stateless detail fetching (ResultsPage).
    When `major_index` is provided, it fetches details without modifying session state.
    """
    session_id = req.get("session_id")
    option = req.get("option", "").strip()
    provided_majors = req.get("majors", [])  # [{name, english_name, score, rationale}, ...]
    major_index_override = req.get("major_index")  # NEW: for stateless detail fetching
    
    sess = _get_session(session_id)
    rag: HybridRAG = sess["rag"]
    lang = sess.get("lang") or "en"
    
    # If majors are provided in request and not already in session, store them
    if provided_majors and not sess.get("recommended_majors"):
        sess["recommended_majors"] = provided_majors
        print(f"[followup] Stored {len(provided_majors)} majors in session from request")
    
    # Use majors from session (either just stored or previously stored)
    recommended_majors = sess.get("recommended_majors", [])
    
    # DEBUG: Show what majors are in session
    print(f"[followup:debug] Session has {len(recommended_majors)} recommended_majors:")
    for i, m in enumerate(recommended_majors[:5]):
        print(f"  [{i}] original_name='{m.get('original_name', 'N/A')}', english_name='{m.get('english_name', 'N/A')}'")
    
    try:
        from translator import UITranslator
        ui_tr = UITranslator()
    except Exception:
        ui_tr = None
    
    # Load extracted majors data (has keywords, courses, eligibility)
    import json
    try:
        with open(EXTRACTED_MAJORS_PATH, "r", encoding="utf-8") as f:
            extracted_majors = json.load(f)
    except Exception:
        extracted_majors = []
    
    # Check if we already have a selected major in session
    selected_major_idx = sess.get("selected_major_idx")
    
    # NEW: If major_index_override is provided, use it for stateless detail fetching
    # This allows ResultsPage to fetch details for all majors without affecting session state
    if major_index_override is not None:
        try:
            override_idx = int(major_index_override)
            print(f"[followup:debug] major_index_override={override_idx}, recommended_majors count={len(recommended_majors)}")
            
            if 0 <= override_idx < len(recommended_majors):
                # Use the override index for THIS REQUEST ONLY (don't modify session)
                selected_major_idx = override_idx
                selected_major = recommended_majors[override_idx]
                # Get ORIGINAL (Hebrew) name for matching with extracted_majors.json
                major_name = selected_major.get("original_name") or selected_major.get("english_name") or "this major"
                english_name = selected_major.get("english_name") or major_name
                
                print(f"[followup:debug] Selected major #{override_idx}: original_name='{major_name}', english_name='{english_name}'")
                print(f"[followup:debug] Available extracted_majors (first 5): {[m.get('original_name') for m in extracted_majors[:5]]}")
                
                # Find full major data - MUST match by original_name (extracted_majors has Hebrew names)
                major_data = None
                for m in extracted_majors:
                    m_original = (m.get("original_name") or "").strip()
                    # Match by original_name ONLY (english_name in extracted_majors is also Hebrew)
                    if m_original == major_name:
                        major_data = m
                        print(f"[followup:debug] ✓ Found matching major data: {m_original}")
                        break
                
                if not major_data:
                    print(f"[followup:error] ✗ Could not find major data for '{major_name}'")
                    print(f"[followup:error] This means recommended major name doesn't match any name in extracted_majors.json")
                
                # Jump directly to detail processing (skip menu logic)
                # Process option 1-4 and return WITHOUT menu
                if option in ["1", "why_good"]:
                    response = f"💡 **{major_name}**\n\n"
                    if major_data and rag.llm:
                        try:
                            example_phrases = {"he": "זה מתאים לך כי...", "ar": "هذا مناسب لك لأن...", "en": "This fits you because..."}
                            example = example_phrases.get(lang, example_phrases["en"])
                            prompt = f"""You are an academic advisor. CRITICAL: Respond ONLY in {lang} language.

Example opening in {lang}: {example}

Student chose: {major_name}
Keywords: {', '.join(major_data.get('keywords', [])[:5])}

Explain in 2-3 sentences why this major is a great fit for them based on their profile.
Be encouraging and specific. Max 60 words.

IMPORTANT: Write your response ONLY in {lang} language."""
                            response += rag.llm.invoke(prompt).content.strip()
                        except:
                            response += ui_tr.tr(lang, f"This major aligns well with your strengths and interests!") if ui_tr else "This major aligns well with your strengths and interests!"
                    else:
                        response += ui_tr.tr(lang, f"This major aligns well with your strengths and interests!") if ui_tr else "This major aligns well with your strengths and interests!"
                    return TextResponse(text=response)
                
                elif option in ["2", "eligibility"]:
                    response = f"📋 **{ui_tr.tr(lang, 'דרישות זכאות') if ui_tr else 'דרישות זכאות'} - {major_name}**\n\n"
                    if major_data and major_data.get("eligibility_rules"):
                        rules = major_data["eligibility_rules"]
                        if lang == "he":
                            response += "**דרישות קבלה:**\n"
                        else:
                            response += "**Admission Requirements:**\n"
                        
                        if rules.get("bagrut_certificate_required"):
                            response += ("• תעודת בגרות נדרשת\n" if lang == "he" else "• Bagrut certificate required\n")
                        if rules.get("bagrut_avg_min"):
                            response += (f"• ממוצע בגרות מינימלי: {rules['bagrut_avg_min']}\n" if lang == "he" else f"• Minimum Bagrut average: {rules['bagrut_avg_min']}\n")
                        if rules.get("psychometric_min") and rules.get("psychometric_min") > 0:
                            response += (f"• פסיכומטרי מינימלי: {rules['psychometric_min']}\n" if lang == "he" else f"• Minimum psychometric score: {rules['psychometric_min']}\n")
                        if rules.get("required_subjects"):
                            subjs = ", ".join(rules["required_subjects"])
                            response += (f"• מקצועות נדרשים: {subjs}\n" if lang == "he" else f"• Required subjects: {subjs}\n")
                        if rules.get("min_units"):
                            response += (f"• מינימום יחידות לימוד: {rules['min_units']}\n" if lang == "he" else f"• Minimum study units: {rules['min_units']}\n")
                        if rules.get("min_grade"):
                            response += (f"• ציון מינימלי נדרש: {rules['min_grade']}\n" if lang == "he" else f"• Minimum required grade: {rules['min_grade']}\n")
                        if rules.get("exceptions_quota_10pct"):
                            response += ("• ניתן לקבל עד 10% תלמידים שאינם עונים על כל התנאים\n" if lang == "he" else "• Up to 10% of students may be accepted without meeting all requirements\n")
                        
                        bagrut_json = rag.bagrut_json or {}
                        from query.bagrut_features import eligibility_flags_from_rules
                        ok, msgs = eligibility_flags_from_rules(rules, bagrut_json)
                        response += "\n" + "─" * 40 + "\n"
                        
                        if ok or not msgs:
                            response += ("✅ **חדשות טובות!** אתה עומד בכל הדרישות!" if lang == "he" else "✅ **Good news!** You meet all requirements!")
                        else:
                            response += (f"⚠️ **דרישות חסרות:**\n" if lang == "he" else f"⚠️ **Missing Requirements:**\n")
                            for msg in msgs[:3]:
                                response += f"  • {msg}\n"
                    else:
                        response += (ui_tr.tr(lang, "No specific requirements listed.") if ui_tr else "No specific requirements listed.")
                    return TextResponse(text=response)
                
                elif option in ["3", "courses"]:
                    response = f"📚 **{ui_tr.tr(lang, 'קורסים לדוגמה') if ui_tr else 'קורסים לדוגמה'} - {major_name}**\n\n"
                    if major_data and major_data.get("sample_courses"):
                        courses = major_data["sample_courses"][:8]
                        for i, course in enumerate(courses, 1):
                            response += f"{i}. {course}\n"
                    else:
                        response += (ui_tr.tr(lang, "Course information not available.") if ui_tr else "Course information not available.")
                    return TextResponse(text=response)
                
                elif option in ["4", "career"]:
                    response = f"🎯 **{ui_tr.tr(lang, 'מסלולי קריירה') if ui_tr else 'מסלולי קריירה'} - {major_name}**\n\n"
                    if major_data and rag.llm:
                        try:
                            example_phrases = {"he": "1. מהנדס תוכנה\n2. מנתח נתונים", "ar": "1. مهندس برمجيات\n2. محلل بيانات", "en": "1. Software Engineer\n2. Data Analyst"}
                            example = example_phrases.get(lang, example_phrases["en"])
                            prompt = f"""You are a career advisor. CRITICAL: Respond ONLY in {lang} language.

Example format in {lang}:
{example}

Major: {major_name}
Keywords: {', '.join(major_data.get('keywords', [])[:5])}

List 4-5 potential career paths for graduates. Be specific and realistic.
Format as numbered list. Max 80 words.

IMPORTANT: Write your response ONLY in {lang} language."""
                            response += rag.llm.invoke(prompt).content.strip()
                        except:
                            response += (ui_tr.tr(lang, "Career information will be provided by your academic advisor.") if ui_tr else "Career information will be provided by your academic advisor.")
                    else:
                        response += (ui_tr.tr(lang, "Career information will be provided by your academic advisor.") if ui_tr else "Career information will be provided by your academic advisor.")
                    return TextResponse(text=response)
        except (ValueError, TypeError):
            pass  # Fall through to normal flow
    
    # STAGE 1: If no major selected yet, validate and process major selection
    if selected_major_idx is None:
        # Check if input is numeric
        if not option.isdigit():
            if lang == "he":
                response = f"אנא הקלד מספר בין 1 ל-{len(recommended_majors)} כדי לבחור מגמה"
            elif lang == "ar":
                response = f"الرجاء إدخال رقم بين 1 و {len(recommended_majors)} لاختيار تخصص"
            else:
                response = f"Please type a number between 1 and {len(recommended_majors)} to select a major"
            return TextResponse(text=response)
        
        # Check if number is in valid range
        idx = int(option) - 1
        if idx < 0 or idx >= len(recommended_majors):
            if lang == "he":
                response = f"המספר חייב להיות בין 1 ל-{len(recommended_majors)}. אנא בחר מספר מהרשימה."
            elif lang == "ar":
                response = f"يجب أن يكون الرقم بين 1 و {len(recommended_majors)}. الرجاء اختيار رقم من القائمة."
            else:
                response = f"Number must be between 1 and {len(recommended_majors)}. Please choose from the list."
            return TextResponse(text=response)
        
        # Valid selection - store and show detail menu
        major_name = recommended_majors[idx].get("original_name") or recommended_majors[idx].get("english_name")
        sess["selected_major_idx"] = idx
        sess["selected_major_name"] = major_name
        
        # Ask which detail they want
        if lang == "he":
            response = f"בחרת: {major_name}\n\nמה תרצה לדעת?\n\n"
            response += "1️⃣ למה זה מתאים לי?\n"
            response += "2️⃣ דרישות קבלה וחוסרים\n"
            response += "3️⃣ קורסים לדוגמה\n"
            response += "4️⃣ מסלולי קריירה\n"
            response += "5️⃣ חזרה לרשימת המגמות"
        else:
            response = f"You selected: {major_name}\n\nWhat would you like to know?\n\n"
            response += "1️⃣ Why is this good for me?\n"
            response += "2️⃣ Eligibility & Requirements\n"
            response += "3️⃣ Sample Courses\n"
            response += "4️⃣ Career Paths\n"
            response += "5️⃣ Back to majors list"
        
        return TextResponse(text=response)
    
    # STAGE 2: Major already selected, now validate and process detail option (1-4)
    if selected_major_idx is None:
        msg = ui_tr.tr(lang, "Please first select a major by typing 1, 2, or 3.") if ui_tr else "Please first select a major by typing 1, 2, or 3."
        return TextResponse(text=msg)
    
    if not recommended_majors or selected_major_idx >= len(recommended_majors):
        msg = ui_tr.tr(lang, "No major selected.") if ui_tr else "No major selected."
        return TextResponse(text=msg)
    
    # VALIDATION: Check if option is numeric
    if not option.isdigit():
        if lang == "he":
            response = "אנא הקלד מספר בין 1 ל-5 כדי לבחור אפשרות"
        elif lang == "ar":
            response = "الرجاء إدخال رقم بين 1 و 5 لاختيار خيار"
        else:
            response = "Please type a number between 1 and 5 to select an option"
        return TextResponse(text=response)
    
    # VALIDATION: Check if number is in valid range (1-5)
    option_num = int(option)
    if option_num < 1 or option_num > 5:
        if lang == "he":
            response = "המספר חייב להיות בין 1 ל-5. אנא בחר אפשרות מהרשימה."
        elif lang == "ar":
            response = "يجب أن يكون الرقم بين 1 و 5. الرجاء اختيار خيار من القائمة."
        else:
            response = "Number must be between 1 and 5. Please choose from the list."
        return TextResponse(text=response)
    
    selected_major = recommended_majors[selected_major_idx]
    major_name = selected_major.get("original_name") or selected_major.get("english_name") or "this major"
    
    # Find full major data from extracted majors
    major_data = None
    for m in extracted_majors:
        if m.get("original_name") == major_name or m.get("english_name") == major_name:
            major_data = m
            break
    
    # Process detail options (1-4)
    if option == "1" or option == "why_good":
        # Explain why this major fits the student
        response = f"💡 **{major_name}**\n\n"
        if major_data:
            if rag.llm:
                try:
                    # Get example phrase in target language
                    example_phrases = {
                        "he": "זה מתאים לך כי...",
                        "ar": "هذا مناسب لك لأن...",
                        "en": "This fits you because..."
                    }
                    example = example_phrases.get(lang, example_phrases["en"])
                    
                    prompt = f"""You are an academic advisor. CRITICAL: Respond ONLY in {lang} language.

Example opening in {lang}: {example}

Student chose: {major_name}
Keywords: {', '.join(major_data.get('keywords', [])[:5])}

Explain in 2-3 sentences why this major is a great fit for them based on their profile.
Be encouraging and specific. Max 60 words.

IMPORTANT: Write your response ONLY in {lang} language."""
                    response += rag.llm.invoke(prompt).content.strip()
                except:
                    response += ui_tr.tr(lang, f"This major aligns well with your strengths and interests!") if ui_tr else "This major aligns well with your strengths and interests!"
            else:
                response += ui_tr.tr(lang, f"This major aligns well with your strengths and interests!") if ui_tr else "This major aligns well with your strengths and interests!"
        
        # Re-show menu
        response += "\n\n" + "─" * 40 + "\n\n"
        if lang == "he":
            response += "מה עוד תרצה לדעת?\n\n"
            response += "1️⃣ למה זה מתאים לי?\n"
            response += "2️⃣ דרישות קבלה וחוסרים\n"
            response += "3️⃣ קורסים לדוגמה\n"
            response += "4️⃣ מסלולי קריירה\n"
            response += "5️⃣ חזרה לרשימת המגמות"
        else:
            response += "What else would you like to know?\n\n"
            response += "1️⃣ Why is this good for me?\n"
            response += "2️⃣ Eligibility & Requirements\n"
            response += "3️⃣ Sample Courses\n"
            response += "4️⃣ Career Paths\n"
            response += "5️⃣ Back to majors list"
        
        return TextResponse(text=response)
    
    elif option == "2" or option == "eligibility":
        # Show eligibility rules and what's missing
        response = f"📋 **{ui_tr.tr(lang, 'דרישות זכאות') if ui_tr else 'דרישות זכאות'} - {major_name}**\n\n"
        
        if major_data and major_data.get("eligibility_rules"):
            rules = major_data["eligibility_rules"]
            
            # FIRST: Show ALL the requirements from JSON
            if lang == "he":
                response += "**דרישות קבלה:**\n"
            else:
                response += "**Admission Requirements:**\n"
            
            # Show bagrut certificate requirement
            if rules.get("bagrut_certificate_required"):
                if lang == "he":
                    response += "• תעודת בגרות נדרשת\n"
                else:
                    response += "• Bagrut certificate required\n"
            
            # Show minimum bagrut average
            if rules.get("bagrut_avg_min"):
                if lang == "he":
                    response += f"• ממוצע בגרות מינימלי: {rules['bagrut_avg_min']}\n"
                else:
                    response += f"• Minimum Bagrut average: {rules['bagrut_avg_min']}\n"
            
            # Show psychometric minimum
            if rules.get("psychometric_min") and rules.get("psychometric_min") > 0:
                if lang == "he":
                    response += f"• פסיכומטרי מינימלי: {rules['psychometric_min']}\n"
                else:
                    response += f"• Minimum psychometric score: {rules['psychometric_min']}\n"
            
            # Show required subjects
            if rules.get("required_subjects"):
                subjs = ", ".join(rules["required_subjects"])
                if lang == "he":
                    response += f"• מקצועות נדרשים: {subjs}\n"
                else:
                    response += f"• Required subjects: {subjs}\n"
            
            # Show minimum units
            if rules.get("min_units"):
                if lang == "he":
                    response += f"• מינימום יחידות לימוד: {rules['min_units']}\n"
                else:
                    response += f"• Minimum study units: {rules['min_units']}\n"
            
            # Show minimum grade in subject
            if rules.get("min_grade"):
                if lang == "he":
                    response += f"• ציון מינימלי נדרש: {rules['min_grade']}\n"
                else:
                    response += f"• Minimum required grade: {rules['min_grade']}\n"
            
            # Show exceptions quota
            if rules.get("exceptions_quota_10pct"):
                if lang == "he":
                    response += "• ניתן לקבל עד 10% תלמידים שאינם עונים על כל התנאים\n"
                else:
                    response += "• Up to 10% of students may be accepted without meeting all requirements\n"
            
            # Show raw eligibility text if available
            if major_data.get("eligibility_raw"):
                response += f"\n**{ui_tr.tr(lang, 'Full Details') if ui_tr else 'Full Details'}:**\n"
                response += f"{major_data['eligibility_raw']}\n"
            
            # SECOND: Check student's Bagrut status
            bagrut_json = rag.bagrut_json or {}
            from query.bagrut_features import eligibility_flags_from_rules
            ok, msgs = eligibility_flags_from_rules(rules, bagrut_json)
            
            response += "\n" + "─" * 40 + "\n"
            
            if ok or not msgs:
                # Good news - they meet all requirements!
                if lang == "he":
                    response += "✅ **חדשות טובות!** אתה עומד בכל הדרישות ויכול להגיש מועמדות ישירות לתוכנית זו!"
                elif lang == "ar":
                    response += "✅ **أخبار سارة!** أنت تستوفي جميع المتطلبات ويمكنك التقديم مباشرة لهذا البرنامج!"
                else:
                    response += "✅ **Good news!** You meet all requirements and can apply directly to this program!"
            else:
                # They're missing some requirements
                if lang == "he":
                    response += f"⚠️ **דרישות חסרות:**\n"
                    for msg in msgs[:3]:
                        response += f"  • {msg}\n"
                    response += f"\n💡 **אבל יש פתרון!**\n"
                    response += "אתה יכול לעשות שנת הכנה במכללת רמת גן שתעזור לך להשלים את הדרישות החסרות ולהתקבל לתוכנית!"
                elif lang == "ar":
                    response += f"⚠️ **المتطلبات المفقودة:**\n"
                    for msg in msgs[:3]:
                        response += f"  • {msg}\n"
                    response += f"\n💡 **لكن هناك حل!**\n"
                    response += "يمكنك القيام بسنة تحضيرية في كلية رمات غان ستساعدك على استكمال المتطلبات المفقودة والقبول في البرنامج!"
                else:
                    response += f"⚠️ **Missing Requirements:**\n"
                    for msg in msgs[:3]:
                        response += f"  • {msg}\n"
                    response += f"\n💡 **But there's a solution!**\n"
                    response += "You can do a preparation year at Ramat Gan College that will help you complete the missing requirements and get accepted into the program!"
        else:
            response += ui_tr.tr(lang, "No specific requirements listed.") if ui_tr else "No specific requirements listed."
        
        # Re-show menu
        response += "\n\n" + "─" * 40 + "\n\n"
        if lang == "he":
            response += "מה עוד תרצה לדעת?\n\n"
            response += "1️⃣ למה זה מתאים לי?\n"
            response += "2️⃣ דרישות קבלה וחוסרים\n"
            response += "3️⃣ קורסים לדוגמה\n"
            response += "4️⃣ מסלולי קריירה\n"
            response += "5️⃣ חזרה לרשימת המגמות"
        else:
            response += "What else would you like to know?\n\n"
            response += "1️⃣ Why is this good for me?\n"
            response += "2️⃣ Eligibility & Requirements\n"
            response += "3️⃣ Sample Courses\n"
            response += "4️⃣ Career Paths\n"
            response += "5️⃣ Back to majors list"
        
        return TextResponse(text=response)
    
    elif option == "3" or option == "courses":
        # Show sample courses
        response = f"📚 **{ui_tr.tr(lang, 'קורסים לדוגמה') if ui_tr else 'קורסים לדוגמה'} - {major_name}**\n\n"
        if major_data and major_data.get("sample_courses"):
            courses = major_data["sample_courses"][:8]  # Show up to 8 courses
            for i, course in enumerate(courses, 1):
                response += f"{i}. {course}\n"
        else:
            response += ui_tr.tr(lang, "Course information not available.") if ui_tr else "Course information not available."
        
        # Re-show menu
        response += "\n\n" + "─" * 40 + "\n\n"
        if lang == "he":
            response += "מה עוד תרצה לדעת?\n\n"
            response += "1️⃣ למה זה מתאים לי?\n"
            response += "2️⃣ דרישות קבלה וחוסרים\n"
            response += "3️⃣ קורסים לדוגמה\n"
            response += "4️⃣ מסלולי קריירה\n"
            response += "5️⃣ חזרה לרשימת המגמות"
        else:
            response += "What else would you like to know?\n\n"
            response += "1️⃣ Why is this good for me?\n"
            response += "2️⃣ Eligibility & Requirements\n"
            response += "3️⃣ Sample Courses\n"
            response += "4️⃣ Career Paths\n"
            response += "5️⃣ Back to majors list"
        
        return TextResponse(text=response)
    
    elif option == "4" or option == "career":
        # Show career paths
        response = f"🎯 **{ui_tr.tr(lang, 'מסלולי קריירה') if ui_tr else 'מסלולי קריירה'} - {major_name}**\n\n"
        if major_data and rag.llm:
            try:
                # Get example phrase in target language
                example_phrases = {
                    "he": "1. מהנדס תוכנה\n2. מנתח נתונים\n3. מנהל מוצר",
                    "ar": "1. مهندس برمجيات\n2. محلل بيانات\n3. مدير منتج",
                    "en": "1. Software Engineer\n2. Data Analyst\n3. Product Manager"
                }
                example = example_phrases.get(lang, example_phrases["en"])
                
                prompt = f"""You are a career advisor. CRITICAL: Respond ONLY in {lang} language.

Example format in {lang}:
{example}

Major: {major_name}
Keywords: {', '.join(major_data.get('keywords', [])[:5])}

List 4-5 potential career paths for graduates. Be specific and realistic.
Format as numbered list. Max 80 words.

IMPORTANT: Write your response ONLY in {lang} language."""
                response += rag.llm.invoke(prompt).content.strip()
            except:
                response += ui_tr.tr(lang, "Career information will be provided by your academic advisor.") if ui_tr else "Career information will be provided by your academic advisor."
        else:
            response += ui_tr.tr(lang, "Career information will be provided by your academic advisor.") if ui_tr else "Career information will be provided by your academic advisor."
        
        # Re-show menu
        response += "\n\n" + "─" * 40 + "\n\n"
        if lang == "he":
            response += "מה עוד תרצה לדעת?\n\n"
            response += "1️⃣ למה זה מתאים לי?\n"
            response += "2️⃣ דרישות קבלה וחוסרים\n"
            response += "3️⃣ קורסים לדוגמה\n"
            response += "4️⃣ מסלולי קריירה\n"
            response += "5️⃣ חזרה לרשימת המגמות"
        else:
            response += "What else would you like to know?\n\n"
            response += "1️⃣ Why is this good for me?\n"
            response += "2️⃣ Eligibility & Requirements\n"
            response += "3️⃣ Sample Courses\n"
            response += "4️⃣ Career Paths\n"
            response += "5️⃣ Back to majors list"
        
        return TextResponse(text=response)
    
    elif option == "5" or option == "back":
        # Clear selected major and return to first list
        sess.pop("selected_major_idx", None)
        sess.pop("selected_major_name", None)
        
        if lang == "he":
            response = "איזו מגמה מעניינת אותך? הקלד את המספר (1, 2, או 3)\n\n"
        elif lang == "ar":
            response = "أي تخصص يهمك؟ اكتب الرقم (1، 2، أو 3)\n\n"
        else:
            response = "Which major interests you? Type the number (1, 2, or 3)\n\n"
        
        # Show the majors list again
        for i, major in enumerate(recommended_majors, 1):
            name = major.get("original_name") or major.get("english_name")
            score = major.get("score", 0)
            match_word = "התאמה" if lang == "he" else "Match"
            response += f"{i}. {name} ({score:.1f}% {match_word})\n"
        
        return TextResponse(text=response)
    
    # If we reach here, option wasn't recognized (should not happen due to validation above)
    if lang == "he":
        response = "אפשרות לא מזוהה. אנא בחר 1, 2, 3, 4, או 5."
    elif lang == "ar":
        response = "خيار غير معروف. الرجاء اختيار 1، 2، 3، 4، أو 5."
    else:
        response = "Unrecognized option. Please select 1, 2, 3, 4, or 5."
    return TextResponse(text=response)


@app.post("/turn/major_info", response_model=TextResponse)
def turn_major_info(req: Dict[str, Any]) -> TextResponse:
    """
    Handle user request for major details after recommendations.
    User can type:
    - A number: "1", "2", "3"
    - A major name: "Computer Science", "מדעי המחשב"
    
    Returns formatted eligibility rules + sample courses in target language.
    """
    session_id = req.get("session_id")
    user_text = req.get("user_text", "").strip()
    recommended_majors = req.get("recommended_majors", [])  # [{name, english_name, score}, ...]
    
    sess = _get_session(session_id)
    rag: HybridRAG = sess["rag"]
    lang = sess.get("lang") or "en"
    
    # Try to match by number first (1-3)
    major_to_show = None
    if user_text.isdigit():
        idx = int(user_text) - 1
        if 0 <= idx < len(recommended_majors):
            major_to_show = recommended_majors[idx]
    
    # If not a number, try matching by name (partial or full)
    if not major_to_show and recommended_majors:
        user_lower = user_text.lower()
        for major in recommended_majors:
            original = (major.get("original_name") or "").lower()
            english = (major.get("english_name") or "").lower()
            if user_lower in original or user_lower in english or original in user_lower or english in user_lower:
                major_to_show = major
                break
    
    if not major_to_show:
        from query.query import _ui_tr
        return TextResponse(text=_ui_tr().tr(lang, "I couldn't find that major. Please type 1, 2, or 3, or the major name."))
    
    # Get full major details
    english_name = major_to_show.get("english_name", "")
    major = rag.get_major_by_english_name(english_name)
    if not major:
        from query.query import _ui_tr
        return TextResponse(text=_ui_tr().tr(lang, f"Sorry, I couldn't find details for {english_name}."))
    
    # Check eligibility
    from query.bagrut_features import eligibility_flags_from_rules
    is_eligible, eligibility_messages = eligibility_flags_from_rules(
        major.get("eligibility_rules", {}),
        rag.bagrut_json or {}
    )
    
    # Format response in target language
    from query.query import _ui_tr
    response_parts = []
    
    # Major name + match score
    original_name = major.get("original_name", english_name)
    score = major_to_show.get("score", 0)
    response_parts.append(f"📚 {original_name} ({score:.1f}% {_ui_tr().tr(lang, 'Match')})")
    response_parts.append("")
    
    # Eligibility status
    if is_eligible:
        response_parts.append(f"✅ {_ui_tr().tr(lang, 'You meet the eligibility requirements!')}")
    else:
        response_parts.append(f"⚠️ {_ui_tr().tr(lang, 'Eligibility Requirements:')}")
        for msg in eligibility_messages[:3]:  # Show top 3 issues
            response_parts.append(f"   • {msg}")
    response_parts.append("")
    
    # Sample courses
    courses = major.get("sample_courses", [])
    if courses:
        response_parts.append(f"📖 {_ui_tr().tr(lang, 'Sample Courses:')}")
        for course in courses[:5]:  # Show up to 5 courses
            response_parts.append(f"   • {course}")
        if len(courses) > 5:
            response_parts.append(f"   {_ui_tr().tr(lang, '...and more')}")
    
    return TextResponse(text="\n".join(response_parts))


@app.get("/major/{major_name}")
def get_major_details(major_name: str, session_id: str):
    """
    Get detailed information about a specific major including:
    - Full description
    - Sample courses
    - Eligibility rules
    - Future opportunities
    """
    sess = _get_session(session_id)
    rag: HybridRAG = sess["rag"]
    
    # Find the major in our data
    major = rag.get_major_by_english_name(major_name)
    if not major:
        raise HTTPException(status_code=404, detail=f"Major '{major_name}' not found")
    
    # Get eligibility check
    from query.bagrut_features import eligibility_flags_from_rules
    is_eligible, eligibility_messages = eligibility_flags_from_rules(
        major.get("eligibility_rules", {}),
        rag.bagrut_json or {}
    )
    
    return {
        "original_name": major.get("original_name", ""),
        "english_name": major.get("english_name", ""),
        "keywords": major.get("keywords", []),
        "sample_courses": major.get("sample_courses", []),
        "eligibility_rules": major.get("eligibility_rules", {}),
        "is_eligible": is_eligible,
        "eligibility_messages": eligibility_messages,
        "future_opportunities": major.get("future_opportunities", []),
    }
