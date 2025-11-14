# query/recommender.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from embeddings.embeddings import get_criteria_keys
from embeddings.person_embeddings import PersonProfile
from query.bagrut_features import eligibility_flags_from_rules

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


@dataclass
class StepRec:
    english_name: str
    original_name: str
    score: float
    short_reason: str
    hint: Optional[str] = None


@dataclass
class FinalRec:
    english_name: str
    original_name: str
    score: float
    rationale: str
    bagrut_alignment_pct: int
    eligibility_summary: Dict[str, Any]


class Recommender:
    """
    Two-phase recommendations:
      - recommend_step(...)  -> chat-friendly suggestions (short reasons, no heavy eligibility)
      - recommend_final(...) -> Top-K with rationale + eligibility summary

    Notes:
      * Person vector is built lazily; if scores aren't set yet, we use neutral 50s.
      * Language is controlled externally via set_language().
    """

    def __init__(
        self,
        majors: List[Dict[str, Any]],
        person,
        bagrut_json=None,
        *,
        w_cosine: float = 0.10,        # Reduced to 0.10 - cosine treats all criteria equally (unwanted)
        w_rubric_align: float = 0.70,  # Increased to 0.70 - WEIGHTED rubric (5x tech, 0.3x social, 0.01x lang) should dominate
        w_bagrut_align: float = 0.20,  # Reduced to 0.20 - bagrut subject match secondary
        eligibility_penalty: float = 0.03, # Reduced from 0.15 to 0.03 - eligibility should influence but not eliminate majors (students can do prep year)
    ):
        self.person = person
        self.majors = majors or []
        self.bagrut_json = bagrut_json
        self.ui_language = "en"  # Default, can be set via set_language()
        self.llm = None  # Optional, can be set via set_llm()
        self._keys = get_criteria_keys()
        # Build person vector lazily to avoid touching person.scores at init
        self._pvec: Optional[np.ndarray] = None

        self.w_cosine = float(w_cosine)
        self.w_rubric_align = float(w_rubric_align)
        self.w_bagrut_align = float(w_bagrut_align)
        self.eligibility_penalty = float(eligibility_penalty)

    # ---- language / binding -------------------------------------------------
    def set_language(self, lang: str):
        self.ui_language = lang or self.ui_language
    
    def set_llm(self, llm):
        """Set the LLM for generating rationales."""
        self.llm = llm

    def bind(self, majors: List[Dict[str, Any]]):
        self.majors = majors or []

    # ---- internal: safe scores + lazy person vector -------------------------
    def _get_scores_dict(self) -> Dict[str, float]:
        # PersonProfile exposes as_dict() - USE THIS FIRST
        if hasattr(self.person, "as_dict") and callable(self.person.as_dict):
            try:
                d = self.person.as_dict()
                if isinstance(d, dict) and d:
                    return d
            except Exception:
                pass

        # Legacy fallback: explicit scores dict (DEPRECATED - PersonProfile uses self.vector)
        if hasattr(self.person, "scores") and isinstance(self.person.scores, dict):
            return self.person.scores

        # Final fallback: neutral 50s per key
        return {k: 50.0 for k in self._keys}


    def _ensure_person_vec(self) -> np.ndarray:
        """
        Build person vector from PersonProfile.as_dict().
        IMPORTANT: Returns values in [0-100] range (NOT normalized to 0-1)!
        This matches the major vectors which are also in [0-100] range.
        """
        # ALWAYS refresh from person.as_dict() to get latest interview updates!
        # (removed caching - Bug #3 fix)
        scores = self._get_scores_dict()
        # Keep values in 0-100 range (DO NOT normalize to 0-1)
        vals = [max(0.0, min(100.0, float(scores.get(k, 50.0)))) for k in self._keys]
        self._pvec = np.array(vals, dtype=float)
        
        # DEBUG: Show ALL scores to debug missing updates
        print(f"[_ensure_person_vec:debug] ALL SCORES from person.as_dict():")
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        for k, v in sorted_scores:
            print(f"  {k}: {v:.1f}")
        print(f"[_ensure_person_vec:debug] Vector average: {np.mean(self._pvec):.1f} (should be 0-100 range)")
        
        return self._pvec

    # ---- similarity & blends -----------------------------------------------
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _rubric_align(self, v: np.ndarray) -> float:
        """
        Calculate WEIGHTED rubric alignment.
        Technical criteria get highest weight (5x).
        Social/psychological criteria get reduced weight (0.3x) to prevent 
        nursing/healthcare from ranking #1 for tech-interested students.
        Language proficiency is essentially ignored (0.01x).
        """
        p = self._ensure_person_vec()
        # Normalize both vectors to [0-1] range
        p_norm = p / 100.0
        v_norm = v / 100.0
        
        # Define criteria weights (aligned with get_criteria_keys() order)
        criteria_keys = get_criteria_keys()
        weights = np.ones(len(criteria_keys))  # Default weight = 1.0
        
        # HIGH PRIORITY (5x weight): Core technical/analytical skills
        high_priority = [
            'data_analysis', 'quantitative_reasoning', 'logical_problem_solving',
            'pattern_recognition', 'systems_thinking', 'theoretical_patience',
            'attention_to_detail'
        ]
        
        # REDUCED (0.3x weight): Social/psychological traits (should be interview-driven)
        # These get high Bagrut seed but shouldn't dominate matching
        social_psychological = [
            'social_sensitivity', 'teamwork', 'communication_mediation',
            'psychological_interest', 'ethical_awareness', 'leadership'
        ]
        
        # IGNORE (0.01x weight): Language proficiency should NOT affect major matching
        low_priority = [
            'hebrew_proficiency', 'english_proficiency',
            'hebrew_expression', 'english_expression'
        ]
        
        for i, key in enumerate(criteria_keys):
            if key in high_priority:
                weights[i] = 5.0  # Technical skills are 5x more important
            elif key in social_psychological:
                weights[i] = 0.3  # Social traits reduced (interview should drive these)
            elif key in low_priority:
                weights[i] = 0.01  # Language effectively ignored (500x less important)
        
        # Calculate WEIGHTED mean absolute difference
        weighted_diff = np.abs(p_norm - v_norm) * weights
        d = weighted_diff.sum() / weights.sum()  # Normalize by total weight
        
        # Convert distance to similarity (1.0 = perfect match, 0.0 = max difference)
        similarity = max(0.0, 1.0 - d)
        
        return similarity

    def _bagrut_align(self, v: np.ndarray) -> float:
        # Dampened reuse of rubric alignment to reflect Bagrut-informed tendencies
        ra = self._rubric_align(v)
        return math.sqrt(max(0.0, ra))

    def _blend(self, v: np.ndarray) -> float:
        """
        Calculate weighted blend of similarity metrics.
        Returns score in PERCENTAGE (0-100) range, not 0-1!
        """
        p = self._ensure_person_vec()
        c = self._cosine(p, v)
        r = self._rubric_align(v)
        b = self._bagrut_align(v)
        
        # Calculate weighted blend (0-1 range)
        blend_normalized = self.w_cosine * c + self.w_rubric_align * r + self.w_bagrut_align * b
        
        # Convert to percentage (0-100 range)
        blend_percentage = blend_normalized * 100.0
        
        return blend_percentage

    def _rank(self) -> List[Tuple[int, float]]:
        """
        Rank all majors by blend score (0-100 percentage).
        DEBUG: Prints detailed breakdown for ALL majors.
        """
        scored: List[Tuple[int, float]] = []
        p = self._ensure_person_vec()
        
        print(f"\n{'='*80}")
        print(f"[_rank] DETAILED SIMILARITY CALCULATIONS FOR ALL MAJORS")
        print(f"Weights: cosine={self.w_cosine:.2f}, rubric={self.w_rubric_align:.2f}, bagrut={self.w_bagrut_align:.2f}")
        print(f"{'='*80}\n")
        
        for i, m in enumerate(self.majors):
            vec = np.array(m.get("vector", []), dtype=float)
            if vec.size != p.size:
                continue
            
            # Calculate all components (0-1 range)
            c = self._cosine(p, vec)
            r = self._rubric_align(vec)
            b = self._bagrut_align(vec)
            
            # Blend returns PERCENTAGE (0-100)
            blend_percentage = self._blend(vec)
            
            # Log ALL majors with basic info
            major_name = m.get('original_name', m.get('english_name', f'Major_{i}'))[:50]
            print(f"[{len(scored)+1:2d}] {major_name:50s} Score={blend_percentage:5.1f}% (cos={c:.3f}, rub={r:.3f}, bag={b:.3f})")
            
            scored.append((i, blend_percentage))
        
        scored.sort(key=lambda t: t[1], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"[_rank] Ranked {len(scored)} majors")
        print(f"{'='*80}\n")
        
        return scored

    # ---- LLM helpers (language via prompt) ----------------------------------
    def _one_liner(self, m: Dict[str, Any]) -> str:
        if not self.llm:
            return f"{m.get('english_name') or m.get('original_name')}"
        prompt = f"""You are an academic advisor. Language: {self.ui_language}
Write ONE sentence (≤20 words) explaining why this major may fit the student.
Major: {m.get('english_name') or m.get('original_name')}
Keywords: {', '.join(m.get('keywords') or [])}
Sample courses: {', '.join(m.get('sample_courses') or [])}
Only output the sentence."""
        return self.llm.invoke(prompt).content.strip().split("\n")[0][:220]

    def _paragraph(self, m: Dict[str, Any], bagrut_pct: int, eligibility_hint: str) -> str:
        if not self.llm:
            return f"Alignment ≈ {bagrut_pct}%. {eligibility_hint}".strip()
        
        # Get person's top strengths for personalization
        person_scores = self._get_scores_dict()
        top_strengths = sorted(person_scores.items(), key=lambda x: -x[1])[:3]
        strengths_text = ", ".join([f"{k.replace('_', ' ')}" for k, _ in top_strengths])
        
        # Parse eligibility to understand admission status
        is_eligible = "eligible" in eligibility_hint.lower() or eligibility_hint == "Eligible."
        
        # Get detailed eligibility rules
        eligibility_rules = m.get('eligibility_rules', {})
        min_grade = eligibility_rules.get('min_grade', 'Not specified')
        required_subjects = eligibility_rules.get('required_subjects', [])
        required_units = eligibility_rules.get('min_units', {})
        
        # Build eligibility details text
        eligibility_details = []
        if min_grade != 'Not specified':
            eligibility_details.append(f"Minimum Bagrut average: {min_grade}")
        if required_subjects:
            eligibility_details.append(f"Required subjects: {', '.join(required_subjects[:3])}")
        if required_units:
            units_text = ', '.join([f"{subj} ({units} units)" for subj, units in list(required_units.items())[:2]])
            if units_text:
                eligibility_details.append(f"Unit requirements: {units_text}")
        
        eligibility_context = " | ".join(eligibility_details) if eligibility_details else eligibility_hint
        
        # ENHANCED PROMPT: Generate comprehensive summary with 5 sections including detailed eligibility
        prompt = f"""You are an academic advisor. CRITICAL: Respond ONLY in {self.ui_language} language.

Generate a comprehensive summary (5-6 sentences) explaining this major to the student.

STRUCTURE YOUR RESPONSE WITH THESE 5 SECTIONS:

1. **Why You're a Great Fit** (1 sentence):
   - Connect their Bagrut strengths to this major's requirements
   - Be specific and encouraging

2. **What You'll Study** (1 sentence):
   - Highlight 2-3 key courses/topics they'll learn
   - Make it concrete and interesting

3. **Skills You'll Gain** (1 sentence):
   - List 3-4 practical skills they'll develop
   - Focus on real-world applications

4. **Admission Requirements** (1-2 sentences) - CRITICAL SECTION:
   - List SPECIFIC requirements: minimum grades, required subjects, units
   - CLEARLY state if student ALREADY MEETS requirements: "✓ You already meet all requirements!" 
   - OR if NOT eligible yet: "You need: [specific missing items]"
   - Be CONCRETE: mention actual grade numbers, subject names, unit counts
   - Examples: 
     * "✓ Eligible now! Requires 80+ Bagrut average (you have 85) and Math 5 units (you have it)."
     * "Requires: Math 5 units (85+), English 4+ units, Bagrut average 75+. You need Math 5 units."

5. **Career Opportunities** (1 sentence):
   - Mention 2-3 specific career paths/job titles
   - Be realistic and inspiring

STUDENT'S PROFILE:
- Top strengths: {strengths_text}
- Bagrut alignment: ~{bagrut_pct}%
- Eligibility status: {"✓ ALREADY ELIGIBLE" if is_eligible else "NEEDS REQUIREMENTS"}

MAJOR DETAILS:
- Name: {m.get('original_name') or m.get('english_name')}
- Keywords: {', '.join((m.get('keywords') or [])[:8])}
- Sample courses: {', '.join((m.get('sample_courses') or [])[:5])}

ADMISSION REQUIREMENTS (mention ALL of these):
{eligibility_context}

CRITICAL REQUIREMENTS:
- Write ONLY in {self.ui_language}
- Section 4 (Admission) MUST be VERY specific with numbers and subject names
- CLEARLY state if student is already eligible (use ✓ or "You're eligible!")
- If not eligible, list EXACTLY what's missing
- Be encouraging but honest
- Total length: 5-6 sentences

Your {self.ui_language} summary:"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            # Ensure response is not empty and is reasonable length
            if response and len(response) > 50:
                return response
            else:
                # Fallback to simpler prompt
                return self._fallback_paragraph(m, bagrut_pct, eligibility_hint, eligibility_rules)
        except Exception as e:
            print(f"[_paragraph:error] LLM generation failed: {e}")
            return self._fallback_paragraph(m, bagrut_pct, eligibility_hint, eligibility_rules)
    
    def _fallback_paragraph(self, m: Dict[str, Any], bagrut_pct: int, eligibility_hint: str, eligibility_rules: Dict[str, Any] = None) -> str:
        """Fallback when LLM fails - generate basic summary."""
        major_name = m.get('original_name') or m.get('english_name')
        courses = ', '.join((m.get('sample_courses') or [])[:3])
        keywords = ', '.join((m.get('keywords') or [])[:4])
        
        # Build admission requirements text
        admission_text = eligibility_hint
        if eligibility_rules:
            min_grade = eligibility_rules.get('min_grade')
            required_subjects = eligibility_rules.get('required_subjects', [])
            
            if min_grade or required_subjects:
                requirements = []
                if min_grade:
                    requirements.append(f"minimum Bagrut average {min_grade}")
                if required_subjects:
                    requirements.append(f"subjects: {', '.join(required_subjects[:2])}")
                
                admission_text = f"Admission requires {' and '.join(requirements)}. {eligibility_hint}"
        
        return f"{major_name} aligns {bagrut_pct}% with your profile. " \
               f"You'll study {courses}. " \
               f"Focus areas include {keywords}. " \
               f"{admission_text}"

    # ---- public APIs --------------------------------------------------------
    def recommend_step(self, top_k: int = 2) -> List[Dict[str, Any]]:
        out: List[StepRec] = []
        for idx, s in self._rank()[:max(1, top_k)]:
            m = self.majors[idx]
            one = self._one_liner(m)
            vec = np.array(m.get("vector", []), dtype=float)
            hint = None
            if self._rubric_align(vec) < 0.7:
                hint = "Clarify what you like more: theory, hands-on labs, or projects."
            out.append(StepRec(
                english_name=m.get("english_name", ""),
                original_name=m.get("original_name", ""),
                score=round(float(s), 4),
                short_reason=one,
                hint=hint
            ))
        return [asdict(x) for x in out]

    def recommend_final(self, top_k: int = 3, bagrut_json: Optional[Dict[str, Any]] = None, interview_text: str = "", degree_filter: str = "both") -> List[Dict[str, Any]]:
        """
        Generate final recommendations with eligibility checks.
        
        Args:
            top_k: Number of recommendations to return
            bagrut_json: Bagrut data for eligibility checking
            interview_text: Combined text from interview answers for field detection
            degree_filter: Filter by degree type - "bachelor", "master", or "both"
        """
        # DEBUG: Show person vector state
        person_scores = self._get_scores_dict()
        top_person_scores = sorted(person_scores.items(), key=lambda x: -x[1])[:5]
        print(f"[recommender:debug] Person top 5 scores:")
        for crit, score in top_person_scores:
            print(f"  {crit}: {score:.1f}")
        
        prelim = self._rank()[:max(1, top_k * 2)]
        
        # DEGREE FILTERING: Filter by bachelor/master degree type
        if degree_filter and degree_filter != "both":
            print(f"[recommender:debug] Filtering by degree type: {degree_filter}")
            filtered_prelim = []
            for idx, base in prelim:
                m = self.majors[idx]
                major_name = m.get('original_name', '')
                
                # Check if major matches requested degree type
                is_bachelor = 'תואר ראשון' in major_name or 'B.A.' in major_name or 'B.Sc.' in major_name
                is_master = 'תואר שני' in major_name or 'M.A.' in major_name or 'M.Sc.' in major_name
                
                if degree_filter == "bachelor" and is_bachelor:
                    filtered_prelim.append((idx, base))
                elif degree_filter == "master" and is_master:
                    filtered_prelim.append((idx, base))
            
            print(f"[recommender:debug] Filtered from {len(prelim)} to {len(filtered_prelim)} majors matching '{degree_filter}'")
            prelim = filtered_prelim
            
            # If we filtered too much, expand the search
            if len(prelim) < top_k:
                print(f"[recommender:debug] Not enough majors after filtering, expanding search...")
                all_ranked = self._rank()
                for idx, base in all_ranked:
                    if len(prelim) >= top_k * 2:
                        break
                    if (idx, base) not in prelim:
                        m = self.majors[idx]
                        major_name = m.get('original_name', '')
                        is_bachelor = 'תואר ראשון' in major_name or 'B.A.' in major_name or 'B.Sc.' in major_name
                        is_master = 'תואר שני' in major_name or 'M.A.' in major_name or 'M.Sc.' in major_name
                        
                        if degree_filter == "bachelor" and is_bachelor:
                            prelim.append((idx, base))
                        elif degree_filter == "master" and is_master:
                            prelim.append((idx, base))
        
        # DEBUG: Show top scored majors BEFORE eligibility
        print(f"[recommender:debug] Top {len(prelim)} majors before eligibility:")
        for idx, base in prelim[:5]:
            m = self.majors[idx]
            print(f"  {m.get('original_name', 'Unknown')}: {base:.3f}")
        
        # FIELD-BASED RERANKING: Detect explicit field mentions and boost/penalize accordingly
        field_adjustments = self._detect_field_preferences(interview_text)
        if field_adjustments:
            print(f"\n[recommend_final:field_boost] Detected field preferences from interview:")
            for field, boost in field_adjustments.items():
                print(f"  {field}: {boost:+.1f}%")
        
        # CHECK FOR NON-TECH STUDENT: If no technical subjects in Bagrut AND no tech mentions,
        # apply penalty to tech majors to avoid recommending CS/IS to language-focused students
        has_tech_subjects = self._check_technical_subjects_from_bagrut(bagrut_json or {})
        has_tech_interest = field_adjustments.get('tech', 0) > 0  # Positive tech boost = interest
        apply_tech_penalty = (not has_tech_subjects) and (not has_tech_interest)
        
        if apply_tech_penalty:
            print(f"\n[recommend_final:tech_penalty] Student has NO tech subjects + NO tech interest → penalizing tech majors by -20%")
        
        adjusted: List[Tuple[int, float, Dict[str, Any]]] = []
        for idx, base in prelim:
            m = self.majors[idx]
            ok, msgs = eligibility_flags_from_rules(m.get("eligibility_rules") or {}, bagrut_json or {})
            penalty = self.eligibility_penalty if (not ok) else 0.0
            
            # Apply field-based adjustment
            field_boost = self._calculate_field_boost(m, field_adjustments)
            
            # Apply tech penalty if student is non-tech
            tech_penalty = 0.0
            if apply_tech_penalty and self._is_tech_major(m):
                tech_penalty = 20.0  # -20% penalty for tech majors
            
            # Calculate final score and CAP at 100 (cannot exceed perfect match)
            # High scores (95+) indicate exceptional fit
            final_score = min(100.0, max(0.0, base - penalty + field_boost - tech_penalty))
            
            # DEBUG: Show calculation for top majors
            major_name = m.get('original_name', '')
            if idx < 5:  # Log first 5
                print(f"\n[recommend_final] {major_name[:40]}")
                print(f"  Base: {base:.1f}%")
                print(f"  Eligibility penalty: -{penalty:.1f}%")
                print(f"  Field boost: {field_boost:+.1f}%")
                print(f"  Tech penalty: -{tech_penalty:.1f}%")
                print(f"  Final: {final_score:.1f}%")
            
            adjusted.append((idx, final_score, {"ok": ok, "msgs": msgs}))
        adjusted.sort(key=lambda t: t[1], reverse=True)

        results: List[FinalRec] = []
        for idx, s, el in adjusted[:max(1, top_k)]:
            m = self.majors[idx]
            vec = np.array(m.get("vector", []), dtype=float)
            bag_pct = int(round(self._rubric_align(vec) * 100))
            elig_hint = "Eligible." if el["ok"] else ("; ".join(el["msgs"][:4]) or "Eligibility unclear.")
            rationale = self._paragraph(m, bag_pct, elig_hint)
            results.append(FinalRec(
                english_name=m.get("english_name", ""),
                original_name=m.get("original_name", ""),
                score=round(float(s), 4),
                rationale=rationale,
                bagrut_alignment_pct=bag_pct,
                eligibility_summary={"pass": el["ok"], "notes": el["msgs"]}
            ))
        return [asdict(x) for x in results]
    
    def _detect_field_preferences(self, interview_text: str) -> Dict[str, float]:
        """
        Detect explicit field mentions in interview text.
        Returns dict of {field_category: boost_percentage}.
        
        Categories:
        - business: כלכלה, עסקים, ניהול, שיווק
        - tech: מחשב, תכנות, אלגוריתם, נתונים
        - health: סיעוד, רפואה, בריאות
        - education: חינוך, הוראה, ייעוץ
        """
        if not interview_text:
            return {}
        
        text_lower = interview_text.lower()
        preferences = {}
        
        # Business/Economics keywords
        business_keywords = ['כלכלה', 'עסק', 'ניהול', 'שיווק', 'מנהל', 'economics', 'business', 'management']
        if any(kw in text_lower for kw in business_keywords):
            preferences['business'] = 8.0   # +8% boost (realistic, not excessive)
            preferences['tech'] = -6.0      # -6% penalty for pure tech
            preferences['health'] = -5.0    # -5% penalty for health
        
        # Tech/CS keywords (if mentioned MORE than business)
        tech_keywords = ['מחשב', 'תכנות', 'אלגוריתם', 'נתונים', 'תוכנה', 'computer', 'programming', 'algorithm', 'software', 
                        'אלקטרוניקה', 'electronics', 'מערכות', 'systems']
        tech_count = sum(1 for kw in tech_keywords if kw in text_lower)
        business_count = sum(1 for kw in business_keywords if kw in text_lower)
        
        if tech_count > business_count:
            preferences['tech'] = 8.0        # +8% boost (realistic)
            preferences['business'] = -6.0   # -6% penalty for business
            preferences['health'] = -8.0     # -8% penalty for health
        
        # Health/Nursing keywords
        health_keywords = ['סיעוד', 'אחיות', 'רפואה', 'בריאות', 'חולה', 'מטופל', 'עזרה לאנשים', 'nursing', 'health', 'medicine', 'patient', 'care']
        has_health_interest = any(kw in text_lower for kw in health_keywords)
        
        if has_health_interest:
            preferences['health'] = 8.0   # +8% boost (realistic)
            preferences['tech'] = -8.0    # -8% penalty for tech
            preferences['business'] = -5.0  # -5% penalty for business
        
        # ANTI-HEALTH: If student shows tech/office interest WITHOUT health keywords, strongly penalize health majors
        office_keywords = ['משרד', 'office', 'מחשב', 'computer', 'אלקטרוניקה', 'electronics']
        has_office_tech_interest = any(kw in text_lower for kw in office_keywords)
        
        if has_office_tech_interest and not has_health_interest:
            # Student wants office/tech work, NOT patient care
            # Apply strong penalty UNLESS student explicitly showed health interest
            current_health = preferences.get('health', 0)
            if current_health <= 0:  # Only apply if health is not positively boosted
                preferences['health'] = -15.0  # Strong penalty - health doesn't fit at all
                print(f"[recommend_final:health_penalty] Student shows office/tech interest WITHOUT health keywords → -15% penalty for health majors")
        
        # Education/Counseling keywords
        education_keywords = ['חינוך', 'הוראה', 'ייעוץ', 'למידה', 'education', 'teaching', 'counseling']
        if any(kw in text_lower for kw in education_keywords):
            preferences['education'] = 8.0  # +8% boost (realistic)
            preferences['tech'] = -6.0      # -6% penalty for tech
        
        return preferences
    
    def _check_technical_subjects_from_bagrut(self, bagrut_json: Dict[str, Any]) -> bool:
        """
        Check if student has ANY technical subjects in Bagrut.
        Technical subjects: Mathematics, Physics, Computer Science, Chemistry
        
        Returns True if found, False otherwise.
        """
        subjects = bagrut_json.get("subjects", [])
        if not subjects:
            return False
        
        technical_subjects = ['מתמטיקה', 'פיזיקה', 'מדעי המחשב', 'כימיה',
                             'mathematics', 'physics', 'computer science', 'chemistry']
        
        for subj in subjects:
            name = (subj.get("name", "") or "").strip().lower()
            if any(tech in name for tech in technical_subjects):
                # Also check if grade is decent (above 60)
                grade = subj.get("grade", 0)
                if grade >= 60:
                    return True
        
        return False
    
    def _is_tech_major(self, major: Dict[str, Any]) -> bool:
        """
        Check if a major is tech-related based on name keywords.
        """
        major_name = (major.get('original_name', '') + ' ' + major.get('english_name', '')).lower()
        tech_keywords = ['מחשב', 'מידע', 'טכנולוגי', 'computer', 'information', 'technology', 'software', 'data']
        return any(kw in major_name for kw in tech_keywords)
    
    def _calculate_field_boost(self, major: Dict[str, Any], field_preferences: Dict[str, float]) -> float:
        """
        Calculate boost/penalty for a major based on detected field preferences.
        
        Maps majors to categories based on name keywords.
        """
        if not field_preferences:
            return 0.0
        
        major_name = (major.get('original_name', '') + ' ' + major.get('english_name', '')).lower()
        boost = 0.0
        
        # Tech majors
        if any(kw in major_name for kw in ['מחשב', 'מידע', 'טכנולוגי', 'computer', 'information', 'technology', 'software']):
            boost += field_preferences.get('tech', 0.0)
        
        # Business majors
        elif any(kw in major_name for kw in ['ניהול', 'כלכלה', 'עסק', 'management', 'economics', 'business']):
            boost += field_preferences.get('business', 0.0)
        
        # Health majors
        elif any(kw in major_name for kw in ['אחיות', 'סיעוד', 'בריאות', 'nursing', 'health']):
            boost += field_preferences.get('health', 0.0)
        
        # Education majors
        elif any(kw in major_name for kw in ['חינוך', 'ייעוץ', 'למידה', 'education', 'counseling', 'learning']):
            boost += field_preferences.get('education', 0.0)
        
        return boost

