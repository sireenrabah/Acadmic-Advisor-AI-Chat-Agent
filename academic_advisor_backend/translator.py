# translator.py
"""
MajorsTranslator — translates original (Hebrew/native) major names to English.

Kept separate so you can change translation strategies (e.g., add a glossary or
post-process names) without reworking extraction or embeddings.
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

_TRANSLATE_TMPL = (
    "You are a precise academic translator. Translate each major name to a conventional English program name.\\n"
    "Return only a JSON object mapping original->english. No commentary.\\n\\n"
    "Examples:\\n"
    "- 'מדעי המחשב' -> 'Computer Science'\\n"
    "- 'הנדסת תעשייה וניהול' -> 'Industrial Engineering and Management'\\n\\n"
    "INPUT (JSON array of major names):\\n{names}\\n\\n"
    "OUTPUT (JSON object):"
)

class MajorsTranslator:
    """Translate a list of original names and return a mapping {original: english}."""

    def __init__(self, model_name: Optional[str] = None):
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        # Temperature 0 for stability/repeatability
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(_TRANSLATE_TMPL)
        self.parser = StrOutputParser()

    def translate_name_map(self, names: List[str]) -> Dict[str, str]:
        """Return a mapping {original_name: english_name}. Falls back to identity on errors."""
        names = [n for n in (names or []) if n]
        if not names:
            return {}
        payload = json.dumps(sorted(list(set(names))), ensure_ascii=False)
        chain = self.prompt | self.llm | self.parser
        out = chain.invoke({"names": payload})
        try:
            mapping = json.loads(out)
            if not isinstance(mapping, dict):
                raise ValueError("Expected a JSON object")
        except Exception:
            mapping = {n: n for n in names}  # identity fallback
        # Ensure coverage for all inputs
        for n in names:
            mapping.setdefault(n, n)
        return mapping

# -------- UI Translator (runtime, cached) --------

_UI_TMPL = (
    "Translate this short UI phrase into {lang}. "
    "Keep it natural, concise, and neutral. "
    "Return only the translation.\n"
    "PHRASE: {t}"
)

class UITranslator:
    """
    Lightweight runtime translator for small UI phrases.
    - Reuses an existing LLM instance if provided.
    - Caches translations per (lang, text) to avoid repeated calls.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI | None = None, model_name: str | None = None):
        if llm is None:
            model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        else:
            self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(_UI_TMPL)
        self.parser = StrOutputParser()
        self._cache: dict[tuple[str, str], str] = {}

    @staticmethod
    def _lang_name(tag_or_name: str) -> str:
        s = (tag_or_name or "").strip().lower()
        if s.startswith("he"): return "Hebrew"
        if s.startswith("ar"): return "Arabic"
        return "English"

    def tr(self, lang: str, text: str) -> str:
        """
        Translate text to target language.
        If lang='en' or 'English', translate FROM any language TO English.
        Otherwise translate TO the specified language.
        """
        if not text:
            return ""
        target = self._lang_name(lang)
        
        # Check cache
        key = (target, text)
        if key in self._cache:
            return self._cache[key]
        
        # Call LLM to translate
        out = (self.prompt | self.llm | self.parser).invoke({"lang": target, "t": text}).strip()
        self._cache[key] = out or text
        return self._cache[key]
