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
    rag = HybridRAG(ui_language=req.language, llm=_LLM)

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
    
    # 2. Simple welcome
    message_parts.append(result['greeting'])
    
    # 3. Bagrut acknowledgment (if available)
    if result.get('bagrut_summary'):
        message_parts.append(result['bagrut_summary'])
    
    # 4. First question
    message_parts.append(result['first_question'])
    
    # Join with double newlines for clear separation
    message = "\n\n".join(message_parts)
    
    return StartResponse(session_id=sid, message=message)

@app.post("/upload")
async def upload(
    session_id: str = Form(...),
    bagrut: Optional[UploadFile] = File(None),
    psychometry: Optional[UploadFile] = File(None),  # not used yet, but accepted
):
    """
    Accepts Bagrut file, saves under /state/uploaded_bagrut.pdf,
    extracts to /state/extracted_bagrut.json, and reseeds the session.
    """
    # Save the uploaded Bagrut (PDF / image)
    if bagrut is not None:
        suffix = ""
        if bagrut.filename:
            suffix = Path(bagrut.filename).suffix or ".pdf"
            # normalize to .pdf target as a single path
        target_path = BAGRUT_UPLOADED_PATH  # keep one canonical filename
        with open(target_path, "wb") as f:
            content = await bagrut.read()
            f.write(content)

        # Run extractor if available
        if BagrutReader is not None:
            try:
                reader = BagrutReader(strict_star_only=True, use_llm_fallback=True, debug=True)
                items, summary = reader.run(str(target_path), out_json=str(BAGRUT_JSON_PATH))
                print(f"[bagrut] extracted {len(items)} items -> {BAGRUT_JSON_PATH.name}")

                # >>> IMPORTANT: make the absolute JSON path visible to all modules/workers
                os.environ["BAGRUT_JSON_PATH"] = str(BAGRUT_JSON_PATH.resolve())

            except Exception as e:
                print(f"[bagrut:error] {e}")
        else:
            if not BAGRUT_JSON_PATH.exists():
                BAGRUT_JSON_PATH.write_text('{"by_subject": {}, "total_units": 0}', encoding="utf-8")
            # Still set it so readers know where to look even with the fallback file
            os.environ["BAGRUT_JSON_PATH"] = str(BAGRUT_JSON_PATH.resolve())

    # Reseed the current session with the freshly extracted JSON
    sess = SESSIONS.get(session_id)
    if sess:
        rag: HybridRAG = sess["rag"]
        lang = sess.get("lang") or "en"
        _seed_rag_with_disk_bagrut(rag, lang)

    # Return success WITHOUT the "Bagrut uploaded & parsed" message
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
        
        # Generate inline recommendations (top 3)
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
