# query/query.py
from __future__ import annotations
import os, json, functools
from typing import Dict, Any, List, Optional, Tuple

from embeddings.embeddings import get_criteria_keys
from embeddings.person_embeddings import PersonProfile, update_person_profile
from query.bagrut_features import (
    bagrut_signals,
    seed_person_vector,
    normalize_bagrut,
    load_bagrut,
    to_canonical_subject_en,
)
from query.recommender import Recommender

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# Use your UI translator for display-only localization (no hard-coded strings)
try:
    from translator import UITranslator
except Exception:
    UITranslator = None

_UI_TR = None
def _ui_tr():
    global _UI_TR
    if _UI_TR is None:
        if UITranslator is not None:
            _UI_TR = UITranslator()
        else:
            class _Noop:
                def tr(self, lang: str, text: str) -> str: return text
            _UI_TR = _Noop()
    return _UI_TR

_KEYS = get_criteria_keys()

def _bagrut_json_path() -> str:
    return os.path.abspath(os.getenv("BAGRUT_JSON_PATH", "state/extracted_bagrut.json"))

def _load_bagrut_from_disk_if_exists() -> Dict[str, Any]:
    p = _bagrut_json_path()
    try:
        if p and os.path.exists(p):
            return load_bagrut(p)
    except Exception:
        pass
    return {}

def _short_or_yesno(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not t: return True
    if len(t.split()) <= 2: return True
    return t in {"yes", "no", "y", "n", "כן", "לא", "ايوا", "لا", "ايوه", "ok", "אוקיי", "אוקי", "طيب"}

def _format_top_subjects_loc(bagrut_json: Dict[str, Any], k: int, ui_lang: str) -> Tuple[str, List[Tuple[str,int,int]]]:
    """
    Return (phrase_for_display, [(canonical_en, grade, units), ...])
    """
    b = normalize_bagrut(bagrut_json or {})
    rows: List[Tuple[str,int,int,int,str]] = []
    for subj_raw, rec in (b.get("by_subject") or {}).items():
        # subj_raw already canonical EN after normalize, but we still pass through translator for UI_lang
        canon_en = to_canonical_subject_en(subj_raw)
        g = int(float(rec.get("final_grade") or 0))
        u = int(rec.get("units") or 0)
        disp = _ui_tr().tr(ui_lang, canon_en)
        rows.append((canon_en, g, u, g * max(1, u), disp))
    rows.sort(key=lambda t: t[3], reverse=True)
    top = rows[:k]
    phrase = ", ".join([f"{disp} ({u}u/{g})" for (_, g, u, _, disp) in top]) if top else ""
    return phrase, [(canon, g, u) for (canon, g, u, _, _) in top]

class HybridRAG:
    """
    No language hard-coding:
      • All UI strings go through Gemini (UITranslator) or the LLM prompt with Language: {lang}.
      • Subjects are canonicalized to English with Gemini, then displayed in the user’s language.
    """

    def __init__(self, ui_language: Optional[str] = None, llm: Optional[ChatGoogleGenerativeAI] = None):
        self.ui_language: Optional[str] = ui_language
        self.llm = llm
        self.degree_filter: str = "both"  # Default: show all degrees

        # PersonProfile uses self.vector (not self.scores)
        self.person = PersonProfile(criteria_keys=_KEYS, llm=llm)

        self.bagrut_json: Dict[str, Any] = {}
        self.signals: Dict[str, float] = {}

        self.majors: List[Dict[str, Any]] = []
        self.vectorstore = None

        self.recommender = Recommender(
            person=self.person,
            majors=[],
        )

        self.last_question_text: str = ""
        self.last_answer_text: str = ""
        self.all_interview_answers: List[str] = []  # Track all answers for field detection

    # ---------- session context ----------
    def set_session_context(self, *, ui_language: Optional[str], bagrut_json: Dict[str, Any]):
        if not ui_language or not isinstance(ui_language, str) or not ui_language.strip():
            raise ValueError("ui_language must be provided via /start.")
        self.ui_language = ui_language.strip()

        self.bagrut_json = dict(bagrut_json or {}) or _load_bagrut_from_disk_if_exists()
        self.signals = bagrut_signals(self.bagrut_json)

        # CRITICAL: Remove old scores dict if it exists (from legacy code)
        # PersonProfile uses self.vector, not self.scores
        if hasattr(self.person, 'scores'):
            delattr(self.person, 'scores')
            print(f"[set_session_context] Removed legacy scores dict")

        # ONLY seed on first call (when person vector is still at neutral 50.0)
        # If person has been updated from interview answers, preserve those updates!
        current_avg = sum(self.person.vector) / len(self.person.vector) if self.person.vector else 50.0
        if abs(current_avg - 50.0) < 1.0:  # Still at neutral seed
            seed = seed_person_vector(_KEYS, self.signals)
            for i, val in enumerate(seed):
                self.person.vector[i] = val
            print(f"[set_session_context] Seeded person vector from Bagrut (avg={current_avg:.1f})")
        else:
            print(f"[set_session_context] Preserving existing person vector (avg={current_avg:.1f} - already updated from interview)")

        self.recommender.set_language(self.ui_language)
        self.recommender.set_llm(self.llm)

    def _ensure_language(self):
        if not self.ui_language:
            raise RuntimeError("Language not set; call /start first.")

    def _has_bagrut(self) -> bool:
        b = normalize_bagrut(self.bagrut_json or {})
        has_subjects = bool((b.get("by_subject") or {}))
        print(f"[_has_bagrut:debug] bagrut_json={self.bagrut_json is not None}, by_subject count={len(b.get('by_subject', {}))}, result={has_subjects}")
        return has_subjects

    def _get_example_phrase(self) -> str:
        """Return native example to prime LLM for target language."""
        examples = {
            "he": "אני רואה שאתה מצטיין במתמטיקה",
            "ar": "أرى أنك متفوق في الرياضيات", 
            "en": "I see you excel in mathematics"
        }
        return examples.get(self.ui_language, examples["en"])

    # ---------- greeting ----------
    def greet_message(self) -> str:
        self._ensure_language()
        # Prefer LLM-crafted greeting in target language
        if self.llm:
            try:
                return self.llm.invoke(
                    f"Write a warm, concise greeting in {self.ui_language}. Max 12 words."
                ).content.strip()
            except Exception:
                pass
        # fallback: translate a simple English greeting
        return _ui_tr().tr(self.ui_language, "Hi! Let's begin your guided interview.")

    # ---------- greeting + first question ----------
    def greet_and_first_question(self) -> Dict[str, Any]:
        """
        Return structured greeting data for 3-part message flow:
        {
            "greeting": str,           # Simple welcome (no Bagrut mention)
            "bagrut_summary": str,     # Highlights TOP 5 strengths with grades
            "first_question": str      # Probes genuine interest
        }
        All in target language.
        """
        self._ensure_language()
        
        if not self._has_bagrut():
            return {
                "greeting": _ui_tr().tr(self.ui_language, "Hi! I'm your academic advisor."),
                "bagrut_summary": "",
                "first_question": _ui_tr().tr(self.ui_language, "Please upload your Bagrut certificate so I can understand your strengths.")
            }
        
        # Get TOP 5 subjects with full details
        subj_phrase, top_subjs = _format_top_subjects_loc(self.bagrut_json, 5, self.ui_language)
        
        if not top_subjs:
            return {
                "greeting": _ui_tr().tr(self.ui_language, "Hi! I'm your academic advisor."),
                "bagrut_summary": "",
                "first_question": _ui_tr().tr(self.ui_language, "Tell me about your interests.")
            }
        
        # 1. GREETING (simple welcome, NO Bagrut mention yet)
        if self.llm:
            try:
                # Build language-specific example first
                example_greeting = {
                    "he": "שלום! שמחה לראות אותך כאן.",
                    "ar": "مرحباً! سعيد برؤيتك هنا.",
                    "en": "Hi there! Great to see you."
                }.get(self.ui_language, "Hi there! Great to see you.")
                
                greeting_prompt = f"""LANGUAGE: {self.ui_language.upper()}
You must respond ONLY in {self.ui_language} language.

Example greeting in {self.ui_language}: "{example_greeting}"

Task: Write a warm 1-sentence welcome greeting for a student.
- Do NOT mention grades or subjects
- Max 12 words
- MUST be in {self.ui_language}

Your {self.ui_language} greeting:"""
                greeting = self.llm.invoke(greeting_prompt).content.strip()
            except Exception:
                greeting = _ui_tr().tr(self.ui_language, "Hi there! Welcome, it's great to see you.")
        else:
            greeting = _ui_tr().tr(self.ui_language, "Hi there! Welcome, it's great to see you.")
        
        # 2. BAGRUT SUMMARY (highlight TOP 5 strengths with grades)
        # Build formatted list of top 5 subjects
        top_5_formatted = []
        units_word = _ui_tr().tr(self.ui_language, "units")
        for name, grade, units in top_subjs:
            name_loc = _ui_tr().tr(self.ui_language, name)
            top_5_formatted.append(f"{name_loc} ({grade}, {units} {units_word})")
        
        top_5_text = ", ".join(top_5_formatted)
        
        if self.llm:
            try:
                # Build language-specific example
                example_bagrut = {
                    "he": f"כל הכבוד על הציונים המרשימים! ראיתי את הציון שלך ב{top_subjs[0][0]} ({top_subjs[0][1]}) ו{top_subjs[1][0]} ({top_subjs[1][1]}) - הישגים מעולים!",
                    "ar": f"تهانينا على درجاتك المذهلة! لاحظت درجتك في {top_subjs[0][0]} ({top_subjs[0][1]}) و{top_subjs[1][0]} ({top_subjs[1][1]}) - إنجازات رائعة!",
                    "en": f"Congratulations on impressive grades! I saw your {top_subjs[0][0]} ({top_subjs[0][1]}) and {top_subjs[1][0]} ({top_subjs[1][1]}) - excellent achievements!"
                }.get(self.ui_language, f"Impressive: {top_5_text}")
                
                bagrut_prompt = f"""LANGUAGE: {self.ui_language.upper()}
You must respond ONLY in {self.ui_language} language.

Example in {self.ui_language}: "{example_bagrut}"

Student's top 5 results: {top_5_text}

Task: Write 1-2 sentences that:
- Acknowledge their achievements (mention 2-3 subjects with grades)
- Show encouragement
- Max 30 words
- MUST be in {self.ui_language}

Your {self.ui_language} acknowledgment:"""
                bagrut_summary = self.llm.invoke(bagrut_prompt).content.strip()
            except Exception:
                bagrut_summary = _ui_tr().tr(self.ui_language, f"Impressive achievements: {top_5_text}! These strong results show dedication.")
        else:
            bagrut_summary = _ui_tr().tr(self.ui_language, f"Impressive achievements: {top_5_text}! These strong results show dedication.")
        
        # 3. FIRST QUESTION (probes genuine interest in TOP subject)
        top_name, top_grade, top_units = top_subjs[0]
        top_name_loc = _ui_tr().tr(self.ui_language, top_name)
        
        if self.llm:
            try:
                # Build language-specific example
                example_question = {
                    "he": f"קיבלת {top_grade} ב{top_name_loc} ({top_units} יחידות) - האם זה באמת מעניין אותך או שזה רק יצא לך בקלות?",
                    "ar": f"حصلت على {top_grade} في {top_name_loc} ({top_units} وحدات) - هل تستمتع به حقاً أم أنه كان سهلاً فقط؟",
                    "en": f"You scored {top_grade} in {top_name_loc} ({top_units} units) - do you genuinely enjoy it, or did it just come naturally?"
                }.get(self.ui_language, f"You scored {top_grade} in {top_name_loc} - do you enjoy it?")
                
                q_prompt = f"""LANGUAGE: {self.ui_language.upper()}
You must respond ONLY in {self.ui_language} language.

Example in {self.ui_language}: "{example_question}"

Student's top subject: {top_name_loc} (grade: {top_grade}, units: {top_units})

Task: Ask ONE conversational question that:
- Mentions the grade and units naturally
- Probes if they genuinely ENJOY the subject (not just excel at it)
- Sounds like a human advisor
- Max 25 words
- MUST be in {self.ui_language}

Your {self.ui_language} question:"""
                first_q = self.llm.invoke(q_prompt).content.strip()
            except Exception:
                first_q = _ui_tr().tr(self.ui_language, f"You scored {top_grade} in {top_name_loc} ({top_units} {units_word}) - do you genuinely enjoy it, or did it just come naturally?")
        else:
            first_q = _ui_tr().tr(self.ui_language, f"You scored {top_grade} in {top_name_loc} ({top_units} {units_word}) - do you genuinely enjoy it, or did it just come naturally?")
        
        return {
            "greeting": greeting,
            "bagrut_summary": bagrut_summary,
            "first_question": first_q  # No prefix needed - frontend strips it anyway
        }

    # ---------- first question (LEGACY - kept for backward compatibility) ----------
    def first_question_after_bagrut(self) -> str:
        """Legacy method - use greet_and_first_question() for better UX."""
        result = self.greet_and_first_question()
        # Store for tracking
        self.last_question_text = result['first_question']
        return f"{result['greeting']}\n\n{result['first_question']}"

    def _build_conversation_context(self, history: List[List[str]], top_subjects: List) -> str:
        """Summarize conversation for LLM context (last 3 turns)."""
        recent = history[-6:] if len(history) >= 6 else history
        lines = []
        for role, text in recent:
            prefix = "Advisor" if role == "ai" else "Student"
            lines.append(f"{prefix}: {text[:100]}")
        return "\n".join(lines) if lines else "(No conversation yet)"

    def _detect_repetitive_answers(self, history: List[List[str]], threshold: int = 3) -> bool:
        """Check if student is giving repetitive short answers (sign to wrap up)."""
        if len(history) < 6:
            return False
        
        # Get last 6 user answers
        user_answers = [text.lower().strip() for role, text in history[-12:] if role == "user"][-6:]
        
        # Count very short/repetitive answers
        short_count = sum(1 for ans in user_answers if len(ans.split()) <= 3)
        
        # Check for repeated words/phrases
        word_counts = {}
        for ans in user_answers:
            for word in ans.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetition = max(word_counts.values()) if word_counts else 0
        
        return short_count >= 4 or max_repetition >= 4

    def _check_vector_convergence(self, threshold: float = 0.05) -> bool:
        """
        Check if PersonProfile vector has stabilized (small recent changes).
        Returns True if the last few updates show minimal movement.
        """
        if not hasattr(self.person, 'history') or len(self.person.history) < 4:
            return False
        
        # Get last 4 updates
        recent_updates = self.person.history[-4:]
        
        # Calculate average delta (absolute change in scores)
        total_delta = 0.0
        count = 0
        for update in recent_updates:
            if 'old' in update and 'new' in update:
                delta = abs(float(update['new']) - float(update['old']))
                total_delta += delta
                count += 1
        
        if count == 0:
            return False
        
        avg_delta = total_delta / count
        
        # If average change is less than threshold (5 points on 0-100 scale), consider converged
        return avg_delta < threshold

    def should_recommend_now(self, *, history: List[List[str]], turn_count: int) -> bool:
        """
        Decide if we should stop asking and show recommendations.
        Uses natural conversation flow based on vector convergence and confidence.
        
        REMOVED hard-coded keyword detection - let LLM's question strategy handle pacing.
        """
        # Stop after 15 turns max (longer conversation for better profiling)
        if turn_count >= 15:
            return True
        
        # Stop if student is being repetitive (early sign of boredom)
        if turn_count >= 8 and self._detect_repetitive_answers(history):
            return True
        
        # Stop if vector has converged (no significant changes)
        if turn_count >= 10 and self._check_vector_convergence():
            return True
        
        # Stop if we have high confidence across criteria
        if hasattr(self.person, 'confidence'):
            avg_confidence = sum(self.person.confidence) / len(self.person.confidence)
            if turn_count >= 10 and avg_confidence >= 0.80:
                return True
        
        return False

    # ---------- next question ----------
    def ask_next_question(self, *, history: List[List[str]], asked_questions: List[str], hint: str = "") -> str:
        self._ensure_language()
        # Always refresh Bagrut in case new upload happened
        self.bagrut_json = _load_bagrut_from_disk_if_exists() or self.bagrut_json
        # Update signals for context, but DON'T re-seed person vector (it has interview updates!)
        self.signals = bagrut_signals(self.bagrut_json)

        if not self._has_bagrut():
            if self.llm:
                q = self.llm.invoke(
                    f"Politely ask in {self.ui_language} for a Bagrut upload; mention you'll tailor questions to strongest subjects/units. 14–18 words."
                ).content.strip()
            else:
                q = _ui_tr().tr(self.ui_language, "Please upload your Bagrut so I can tailor questions to your strengths.")
            self.last_question_text = q
            return q  # No prefix needed

        recent_tail = " • ".join(asked_questions[-3:]) if asked_questions else "none"
        subj_phrase, top = _format_top_subjects_loc(self.bagrut_json, 3, self.ui_language or "en")

        if _short_or_yesno(self.last_answer_text):
            guidance = "Ask for one concrete preference with a tiny example (course/project) they imagine enjoying."
        else:
            guidance = "Narrow toward majors with contrasting choices."

        # Build conversation context
        conversation_context = self._build_conversation_context(history, top)

        if self.llm:
            # Build language-specific example question
            example_next_q = {
                "he": "ראיתי שאתה חזק במתמטיקה - האם אתה מעדיף לעבוד בצוות או באופן עצמאי?",
                "ar": "لاحظت أنك قوي في الرياضيات - هل تفضل العمل في فريق أم بشكل مستقل؟",
                "en": "I saw you're strong in Math - do you prefer working in a team or independently?"
            }.get(self.ui_language, "Do you prefer working alone or in teams?")
            
            prompt = f"""LANGUAGE: {self.ui_language.upper()}
You must respond ONLY in {self.ui_language} language.

Example question in {self.ui_language}: "{example_next_q}"

ROLE: Academic advisor for HIGH SCHOOL GRADUATE (17-18 years old)

STUDENT'S TOP STRENGTHS: {subj_phrase}

CONVERSATION SO FAR:
{conversation_context}

IMPORTANT CONTEXT:
- This is a YOUNG ADULT who just finished high school
- They DON'T know deep technical details about fields (NLP, machine learning, etc.)
- They ARE exploring what TYPE OF WORK suits their personality

YOUR QUESTIONING STRATEGY:
Turn 1-2: Ask about Bagrut subjects - which did they genuinely ENJOY?
Turn 3-4: Ask about WORK ENVIRONMENT preferences:
  - Indoors (office/lab) or outdoors?
  - Independently or in teams?
  - Physical work (hands-on) or mental work (thinking)?
  - Working with people or with systems/data?
Turn 5-6: Ask about DAILY ACTIVITIES they find satisfying:
  - Solving puzzles vs. helping people vs. creating things?
  - Structured routine vs. varied challenges?

AVOID:
- ❌ Technical jargon (algorithms, NLP, data structures)
- ❌ Research topics (they're choosing BACHELOR'S, not PhD)
- ❌ Over-specific sub-fields

DO:
- ✅ Ask about personality, work style, preferences
- ✅ Use concrete examples: "hospital" vs. "office"
- ✅ Reference their Bagrut naturally

RECENTLY ASKED (do NOT repeat):
{chr(10).join(f"- {q}" for q in (asked_questions[-2:] if len(asked_questions) >= 2 else []))}

HINT: {hint or guidance}

Requirements:
- ONE question only
- 1-2 sentences max
- Natural conversational {self.ui_language}
- Age-appropriate (high school level)
- MUST be in {self.ui_language}

Your {self.ui_language} question:"""
            q = self.llm.invoke(prompt).content.strip()
        else:
            base = f"With {subj_phrase} in mind, pick one area to explore and why."
            q = _ui_tr().tr(self.ui_language, base)

        # Deduplicate if Gemini echoed same wording
        if self.last_question_text and q.strip() == self.last_question_text.strip():
            base = f"Considering {subj_phrase}, which suits you now—data/programming, lab/biomed, or policy/language—and why?"
            q = _ui_tr().tr(self.ui_language, base)

        self.last_question_text = q
        return q  # No prefix needed

    # ---------- absorb ----------
    def absorb_answer(self, *, user_text: str, last_question: str) -> bool:
        """
        Absorb user's answer and update PersonProfile vector using LLM-based analysis.
        The LLM identifies which criteria are relevant and scores them 0-100.
        """
        import sys
        self.last_answer_text = (user_text or "").strip()
        
        # Track all answers for field detection in recommendations
        if self.last_answer_text:
            self.all_interview_answers.append(self.last_answer_text)
        
        try:
            # DEBUG: Print vector state before update
            old_scores = self.person.as_dict().copy()
            
            print(f"\n{'='*80}", flush=True)
            print(f"[absorb:start] Processing answer: '{user_text[:50]}...'", flush=True)
            print(f"[absorb:start] Last question was: '{last_question[:80]}...'", flush=True)
            print(f"{'='*80}\n", flush=True)
            sys.stdout.flush()
            
            # Detect Hebrew technical keywords as HINTS for the LLM (not as primary signals)
            # This helps guide the LLM but doesn't bypass it
            detected_keywords = []
            answer_lower = user_text.lower()
            
            # Simple keyword detection (typo-tolerant via substring matching)
            keyword_hints = [
                ("מתמטיקה", "mathematics"),
                ("מתמתיקה", "mathematics"),
                ("הסתברות", "probability"),
                ("אלגוריתם", "algorithms"),  # Catches both spellings
                ("אלגורתם", "algorithms"),  # Alternative spelling
                ("ניתוח נתונים", "data analysis"),
                ("נתונים", "data"),
                ("תכנות", "programming"),
                ("מחשב", "computer science"),
                ("פתר", "problem solving"),  # Catches פתרון/פתירת
                ("בעיות", "problems"),
            ]
            
            for hebrew_word, english_hint in keyword_hints:
                if hebrew_word in answer_lower:
                    detected_keywords.append(english_hint)
            
            # Detect passion/interest words
            passion_words = ["אוהב", "אוהבת", "מעניין", "מעניינת", "נהנה", "נהנית"]
            has_passion = any(word in answer_lower for word in passion_words)
            
            if detected_keywords:
                print(f"[absorb:keywords] Detected hints: {', '.join(detected_keywords)}")
                if has_passion:
                    print(f"[absorb:keywords] Student expressed passion/interest")
            
            # Use LLM as PRIMARY analysis method (typo-tolerant, context-aware)
            if self.llm:
                criteria_list = "\n".join([f"- {k}" for k in _KEYS])
                
                # Get current top 5 scores for context
                top_current = sorted(old_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                current_context = "\n".join([f"- {k}: {v:.1f}" for k, v in top_current])
                
                # Build keyword hints section if detected
                keyword_context = ""
                if detected_keywords:
                    keyword_context = f"\n\nDETECTED KEYWORDS IN ANSWER: {', '.join(detected_keywords)}"
                    if has_passion:
                        keyword_context += " (with passion/interest expressed)"
                
                analysis_prompt = f"""You are analyzing a student's answer to identify their interests and skills.

QUESTION: {last_question}
ANSWER: {user_text}{keyword_context}

CURRENT TOP SCORES (from Bagrut/previous answers):
{current_context}

AVAILABLE CRITERIA (choose up to 3 most relevant):
{criteria_list}

SCORING GUIDELINES:
- Mathematics/probability → quantitative_reasoning, logical_problem_solving, pattern_recognition
- Algorithms/programming/coding → logical_problem_solving, systems_thinking, data_analysis
- Data science/statistics/analysis → data_analysis, quantitative_reasoning, pattern_recognition
- Computer science topics → data_analysis, logical_problem_solving, systems_thinking
- Passion words (אוהב, love, enjoy, interesting, מעניין, נהנה) → boost to 96-100 range
- Technical terms WITHOUT passion → score 88-94
- Moderate interest → score 75-87
- Weak interest → score 50-70

CRITICAL: If student mentions BOTH technical term AND passion word, score 96-100!
CRITICAL: "Algorithms" (אלגוריתמים/אלגורתמים) should ALWAYS include data_analysis + logical_problem_solving + systems_thinking!

Return ONLY a JSON object with 1-3 most relevant criteria:
{{"criterion_name": score_0_to_100}}

JSON:"""
                
                try:
                    response = self.llm.invoke(analysis_prompt).content.strip()
                    # Extract JSON
                    import json, re
                    json_match = re.search(r'\{[^}]+\}', response)
                    if json_match:
                        criteria_scores = json.loads(json_match.group(0))
                        
                        # DEBUG: Show what LLM identified
                        print(f"[absorb:llm] LLM response: {criteria_scores}")
                        
                        # Update profile with identified criteria
                        for criterion, score in criteria_scores.items():
                            if criterion in _KEYS:
                                self.person.update_from_answer(
                                    criterion,
                                    float(score),
                                    note="llm_analysis",
                                    ext_weight=1.0,
                                    temp=0.0
                                )
                        
                        print(f"[absorb:llm] Identified {len(criteria_scores)} relevant criteria from answer")
                    else:
                        print(f"[absorb:warn] LLM response not valid JSON: {response[:100]}")
                        # Fallback to old method
                        self._fallback_absorb(user_text, last_question)
                        
                except Exception as e:
                    print(f"[absorb:error] LLM analysis failed: {e}")
                    self._fallback_absorb(user_text, last_question)
            else:
                # No LLM - use fallback
                self._fallback_absorb(user_text, last_question)
            
            # DEBUG: Print vector state after update
            new_scores = self.person.as_dict()
            changed_criteria = {k: (old_scores[k], new_scores[k]) 
                              for k in old_scores.keys() 
                              if abs(old_scores[k] - new_scores[k]) > 1.0}
            if changed_criteria:
                print(f"[absorb] Vector updated: {len(changed_criteria)} criteria changed")
                for k, (old, new) in list(changed_criteria.items())[:5]:  # Show top 5 changes
                    print(f"  {k}: {old:.1f} → {new:.1f} (Δ{new-old:+.1f})")
            else:
                print(f"[absorb:warn] No criteria changed (answer may be too vague)")
            
            if not hasattr(self.person, "scores") or not isinstance(self.person.scores, dict):
                self.person.scores = {k: 50.0 for k in _KEYS}
            return True
        except Exception as e:
            print(f"[absorb:error] {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _fallback_absorb(self, user_text: str, last_question: str):
        """Fallback method when LLM not available - uses person_embeddings.py logic."""
        update_person_profile(
            self.person,
            question_id=last_question,
            answer_text=user_text,
            history_text="",
            weight=1.0,
            temperature=0.0,
        )

    # ---------- recommendations ----------
    def transition_to_recommendations(self) -> str:
        """Natural message before showing results."""
        if not self.llm:
            return _ui_tr().tr(self.ui_language, "Based on your answers, here are my recommendations:")
        
        subj_phrase, _ = _format_top_subjects_loc(self.bagrut_json, 2, self.ui_language)
        
        try:
            prompt = f"""You are an academic advisor in {self.ui_language}.
The student has strong Bagrut results in: {subj_phrase}
You just finished interviewing them about interests.

Write a warm 2-sentence transition saying:
1. You've learned about their interests
2. You're now showing matched majors

Natural advisor tone. Max 25 words. Language: {self.ui_language}"""
            
            return self.llm.invoke(prompt).content.strip()
        except Exception:
            return _ui_tr().tr(self.ui_language, "Based on your answers, here are my recommendations:")

    def format_inline_recommendations(self, *, top_k: int = 3) -> str:
        """
        Generate formatted top-K recommendations for inline chat display.
        Returns a string with numbered list of majors + short reasons.
        """
        print(f"[recommendations] Generating with {len(self.majors)} majors loaded")
        print(f"[recommendations] Person vector: {len(self.person.as_dict())} criteria")
        print(f"[recommendations] Degree filter: {self.degree_filter}")
        
        self.recommender.bind(self.majors)
        
        # Combine all interview answers for field detection
        interview_text = " ".join(self.all_interview_answers)
        recs = self.recommender.recommend_final(
            top_k=top_k, 
            bagrut_json=self.bagrut_json,
            interview_text=interview_text,
            degree_filter=self.degree_filter  # Pass degree filter
        )
        
        print(f"[recommendations] Got {len(recs)} recommendations")
        if recs:
            for i, rec in enumerate(recs[:3], 1):
                print(f"  {i}. {rec.get('english_name')} - score: {rec.get('score', 0):.1f}")
        
        if not recs:
            return _ui_tr().tr(self.ui_language, "I couldn't find suitable matches. Please upload your Bagrut or answer more questions.")
        
        # Build formatted list
        lines = []
        for i, rec in enumerate(recs, 1):
            # Get major name in target language
            major_name = rec.get('original_name', rec.get('english_name', 'Unknown'))
            score = rec.get('score', 0)
            rationale = rec.get('rationale', '')
            
            # Translate "Match" to target language
            match_word = _ui_tr().tr(self.ui_language, "Match")
            
            # Format: "1. Computer Science (85.2% Match)\n   Strong programming skills align with your Math strength."
            lines.append(f"{i}. {major_name} ({score:.1f}% {match_word})")
            if rationale:
                # Limit rationale to first sentence for brevity
                first_sentence = rationale.split('.')[0].strip()
                if first_sentence:
                    lines.append(f"   {first_sentence}.")
        
        # Add follow-up question about viewing eligibility/courses
        lines.append("")  # Blank line for spacing
        follow_up = _ui_tr().tr(
            self.ui_language,
            "Would you like to see the eligibility requirements or sample courses for any of these majors?"
        )
        lines.append(follow_up)
        
        return "\n".join(lines)

    def recommend_step(self, *, top_k: int = 2) -> List[Dict[str, Any]]:
        self.recommender.bind(self.majors)
        return self.recommender.recommend_step(top_k=top_k)

    def recommend_final(self, *, top_k: int = 3) -> List[Dict[str, Any]]:
        self.recommender.bind(self.majors)
        interview_text = " ".join(self.all_interview_answers)
        return self.recommender.recommend_final(
            top_k=top_k, 
            bagrut_json=self.bagrut_json,
            interview_text=interview_text,
            degree_filter=self.degree_filter  # Pass degree filter
        )

    # ---------- majors I/O ----------
    def load_majors_metadata(self, path: str) -> int:
        if not os.path.exists(path):
            self.majors = []
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.majors = data if isinstance(data, list) else []
        return len(self.majors)

    def load_majors_rubric_vectors(self, path: str, eligibility_rules_path: str) -> int:
        vectors: Dict[str, List[float]] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and isinstance(raw.get("vectors"), list):
                for row in raw["vectors"]:
                    try:
                        en = row[0]; vec = [float(x) for x in row[1]]
                        vectors[en] = vec
                    except Exception:
                        pass
            elif isinstance(raw, list):
                for row in raw:
                    try:
                        en = row.get("english_name")
                        vec = [float(x) for x in (row.get("vector") or [])]
                        if en and vec:
                            vectors[en] = vec
                    except Exception:
                        pass
        for m in self.majors:
            en = (m.get("english_name") or "").strip()
            if en and en in vectors:
                m["vector"] = vectors[en]
        return len(vectors)

    def get_major_by_english_name(self, en: str) -> Dict[str, Any]:
        for m in self.majors:
            if (m.get("english_name") or "").strip() == (en or "").strip():
                return m
        return {}
