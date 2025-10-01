# generation/generate.py
from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional, Dict, Set

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None

from embeddings import get_criteria, get_criteria_keys


class QuestionGenerator:
    """
    Generates ONE interview turn:
      - Returns 'QUESTION: ... [criterion=<key>]' for rubric-targeted probes
      - First turn: asks for preferred language (no LLM call)
      - Uses the saved self.preferred_language for all later turns
    """

    def __init__(
        self,
        llm: Optional[object] = None,
        llm_model: str = "gemini-2.0-flash",
        temperature: float = 0.2,
    ):
        self.llm = llm or self._make_llm(llm_model, temperature)
        self._criteria_list: List[Tuple[str, str]] = get_criteria()
        self._criteria_keys: List[str] = get_criteria_keys()
        self._criteria_map: Dict[str, str] = {k: d for k, d in self._criteria_list}

        # NEW: will be set right after the first prompt is answered
        self.preferred_language: Optional[str] = None

    # ---------- LLM helper ----------

    def _make_llm(self, model_name: str, temperature: float):
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain_google_genai is not installed.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)

    # ---------- Utilities ----------

    @staticmethod
    def _extract_asked_criteria(asked_questions: List[str]) -> Set[str]:
        seen: Set[str] = set()
        rx = re.compile(r"\[criterion=([a-zA-Z0-9_]+)\]")
        for q in asked_questions or []:
            m = rx.search(q)
            if m:
                seen.add(m.group(1))
        return seen

    @staticmethod
    def _last_user_message(history: List[Tuple[str, str]]) -> str:
        """Grab the latest user message text from history ('' if none)."""
        for role, text in reversed(history or []):
            if role == "user" and isinstance(text, str):
                return text.strip()
        return ""

    def _pick_next_criterion(
        self,
        asked_questions: Optional[List[str]],
        current_scores: Optional[Dict[str, int]],
    ) -> Tuple[str, str]:
        """
        Strategy:
          1) Prefer a facet that hasn't been asked yet (from [criterion=...] tags)
          2) Otherwise pick the facet with the lowest current score
          3) Fallback to the first criterion
        """
        asked_set = self._extract_asked_criteria(asked_questions or [])
        # 1) untouched facets first
        for k in self._criteria_keys:
            if k not in asked_set:
                return k, self._criteria_map.get(k, "")
        # 2) min-score facet
        if current_scores:
            keys_sorted = sorted(self._criteria_keys, key=lambda k: float(current_scores.get(k, 0)))
            k = keys_sorted[0]
            return k, self._criteria_map.get(k, "")
        # 3) default
        k = self._criteria_keys[0]
        return k, self._criteria_map.get(k, "")

    # ---------- Public: one turn ----------

    def interview_turn(
        self,
        history: List[Tuple[str, str]],
        majors_hint: str = "",
        asked_questions: Optional[List[str]] = None,
        force_question: bool = False,
        avoid_repeat_of: Optional[str] = None,
        current_scores: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Build the prompt and return a single line that starts with:
        • 'QUESTION:' (preferred)
        • or 'RECOMMEND:' if the model is confident to stop
        The rubric-targeted question MUST end with ' [criterion=<key>]'.

        Notes:
        - Turn 1: ask for preferred language (no LLM call).
        - On the next call, we SAVE the user's reply into self.preferred_language.
        - No automatic language detection.
        """
        print("Welcome To The Acadmic Advisor!")

        asked_questions = asked_questions or []

        # ---- FIRST TURN: explicit language selection (no LLM call needed) ----
        if not asked_questions:
            return "Please type your preferred interview language (e.g., English): "

        # ---- If language not saved yet, grab it from the latest user message and save ----
        if self.preferred_language is None:
            reply = self._last_user_message(history)
            # keep it simple: take the first line, trim, cap length a bit
            self.preferred_language = (reply.splitlines()[0] if reply else "English").strip()[:64] or "English"

        preferred_lang = self.preferred_language or "English"

        # Which rubric facet to probe next?
        target_key, target_desc = self._pick_next_criterion(asked_questions, current_scores)

        # Coverage status for the model
        asked_set = self._extract_asked_criteria(asked_questions)
        cov = f"Probed facets so far: {len(asked_set)}/{len(self._criteria_keys)}."

        # System instructions (no language detection; just use preferred_lang)
        system = (
            "You are an academic advisor interviewing a student. "
            "Ask deep, diagnostic questions ONE at a time to infer the student's strengths and preferences "
            "on a defined rubric. Avoid superficial prompts like 'what is your favorite subject'.\n\n"
            "GOAL:\n"
            "- Elicit evidence to rate the rubric facets (0–100) by probing scenarios, trade-offs, past behavior, and constraints.\n"
            "- Prefer questions that force reflection: choices under constraints, teamwork conflicts, leadership/ethics dilemmas, etc.\n\n"
            "LANGUAGE POLICY:\n"
            f"- The student's chosen language is: {preferred_lang}.\n"
            "- Always reply ONLY in that language.\n"
            "- Do not repeat questions you've already asked.\n\n"
            "OUTPUT:\n"
            "- Return exactly one line beginning with 'QUESTION:' "
            "(or 'RECOMMEND:' if you have enough information to suggest 3 majors).\n"
            "- Append a tag ' [criterion=<key>]' to indicate which rubric facet this question targets.\n"
        )
        if majors_hint:
            system += f"Relevant majors set (optional hint for your mental model): {majors_hint}. "
        if force_question:
            system += "IMPORTANT: Output a single line that begins with 'QUESTION:' only. "
        if avoid_repeat_of:
            system += f"Do NOT ask anything similar to this: '{avoid_repeat_of}'. "

        asked_blob = "; ".join(asked_questions[-10:]) if asked_questions else "none"
        rubric_lines = "\n".join([f"- {k}: {d}" for k, d in self._criteria_list])

        # Keep conversation context (helps tone/continuity)
        chat_msgs = [("system", system)]
        for role, text in history:
            chat_msgs.append(("ai" if role == "ai" else "human", text))

        # Instruction block (embed target variables directly; only {rubric} stays as template var)
        chat_msgs.append((
            "human",
            "Rubric facets (key → meaning):\n{rubric}\n\n"
            f"Already asked (last few): {asked_blob}\n"
            f"{cov}\n"
            "Your task now:\n"
            f"- Target facet: {target_key} — {target_desc}\n"
            "- Produce ONE probing question that elicits reasoning and (optionally) a 0–100 self-estimate.\n"
            f"- The line must start with 'QUESTION:' and end with ' [criterion={target_key}]'."
        ))

        if ChatPromptTemplate is None:
            raise RuntimeError("langchain-core is not installed.")
        chat = ChatPromptTemplate.from_messages(chat_msgs)
        chain = chat | self.llm

        out = chain.invoke({"rubric": rubric_lines}).content.strip()
        return out

    # ---------- Public: fallback (rule-based) ----------

    @staticmethod
    def fallback_question() -> str:
        """
        Simple, safe fallback if the LLM call fails or repeats.
        Uses a high-signal facet: logical_problem_solving.
        """
        return (
            "QUESTION: You're given ambiguous requirements and a tight deadline. "
            "How would you break the problem down and choose a first step? "
            "Please include a brief example and (optionally) a 0–100 self-rating. "
            "[criterion=logical_problem_solving]"
        )
