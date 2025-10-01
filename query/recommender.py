# query/recommender.py
from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple, Any

# LangChain bits for explanation generation (optional)
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None


class Recommender:
    """
    Owns recommendation logic over ~50-dim rubric vectors (0..100):
      • Cosine similarity scoring (student vector vs. major vectors)
      • Optional explanation generation from your vectorstore + LLM
      • Final block formatting for the interview flow

    Construct with the tools it needs (embeddings, vectorstore, llm, output_parser),
    then bind majors/vectors via bind_majors(...).
    """

    def __init__(self, embeddings, vectorstore, llm, output_parser):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.llm = llm
        self.output_parser = output_parser

        # In-memory data bound later
        self.majors: List[Any] = []  # your domain objects / dicts for majors
        # (english_name, rubric_vector_0_100_aligned_to_keys)
        self._major_vectors: List[Tuple[str, List[float]]] = []
        # english_name -> original_name (for better retrieval/display)
        self._name_map: Dict[str, str] = {}

    # ---------------------- majors binding ----------------------

    def bind_majors(
        self,
        majors: List[Any],
        major_vectors: List[Tuple[str, List[float]]],
        name_map: Dict[str, str],
    ) -> None:
        """
        Call this after you load majors and precomputed vectors in HybridRAG.

        Args:
            majors: your majors list (objects/dicts, used for hints)
            major_vectors: [(english_name, rubric_vector_0_100), ...]
            name_map: {english_name: original_name}
        """
        self.majors = majors or []
        self._major_vectors = major_vectors or []
        self._name_map = name_map or {}

    # ---------------------- recommendation core ----------------------

    def recommend_from_interview(
        self,
        person_vector: List[float],
        top_k: int = 3,
    ) -> List[Tuple[str, float, str]]:
        """
        Take the student's rubric vector (0..100, same key order as majors),
        compute cosine similarity against each major vector, and return top_k:
        [(english_major_name, cosine_score, rationale_text), ...]

        NOTE: self._major_vectors must be [(english_name, vector_0..100), ...]
        """
        if not self._major_vectors:
            return []

        # --- coerce incoming vector to expected length & range 0..100 ---
        target_len = len(self._major_vectors[0][1])
        pv = list(person_vector or [])
        pv = [float(0 if x is None else max(0.0, min(100.0, float(x)))) for x in pv]
        if len(pv) < target_len:
            pv = pv + [0.0] * (target_len - len(pv))
        elif len(pv) > target_len:
            pv = pv[:target_len]

        # --- cosine helper ---
        def _cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

        # --- score all majors ---
        sims = [(name, _cosine(pv, mvec)) for name, mvec in self._major_vectors]
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:max(1, int(top_k))]

        # --- lightweight profile text for the explainer prompt (optional) ---
        profile_text = "Student rubric vector (0–100) used for similarity matching."

        # --- attach rationales (uses vectorstore+LLM if available) ---
        results: List[Tuple[str, float, str]] = []
        for name, sc in top:
            original = self._name_map.get(name, name)
            why = self._explain_major_choice(name, original, profile_text)
            results.append((name, float(sc), why))
        return results

    def final_block_from_person_vector(self, person_vector: List[float], top_k: int = 3) -> str:
        """
        Format final recommendations from a 0–100 rubric vector.
        This is the entry the interview flow should call.
        """
        recs = self.recommend_from_interview(person_vector, top_k=top_k)
        if not recs:
            return "[warn] Could not compute recommendations. Ensure majors & vectors are bound via bind_majors(...)."

        lines = ["\n--- Final Recommendations (Ramat Gan Academic College Majors Only) ---"]
        lines.append("-------------------------------------------------------------------")
        
        for i, (name_en, score, why) in enumerate(recs, start=1):
            # Add an empty line between recommendations (not before the first one)
            if i > 1:
                lines.append("")  

            lines.append(f"{i}. {name_en}  (match {score*100:.1f}%)")
            if why:
                lines.append(f"   • Why: {why}")

        return "\n".join(lines)


    # ------------------ context & explanation (optional) ------------------

    def _retrieve_major_context(
        self,
        english_name: str,
        original_name: Optional[str],
        extra_hint: str = "",
    ) -> List[Any]:
        """
        Pull a few chunks from your vectorstore to help the LLM write a rationale.
        Returns [] if vectorstore is unavailable.
        """
        if not self.vectorstore:
            return []
        q = " ".join(filter(None, [english_name, original_name, extra_hint]))
        try:
            return [d for d in self.vectorstore.similarity_search(q, k=4)]
        except Exception:
            return []

    def _explain_major_choice(
        self,
        english_name: str,
        original_name: str,
        student_profile_text: str,
    ) -> str:
        """
        Produces 2–3 sentence rationale using (optional) PDF context and LLM.
        Falls back to a static message if prompt tools aren't available.
        """
        # small hint from the bound majors list (keywords/courses)
        hint = ""
        for m in self.majors:
            # works with dicts or objects
            en = m.get("english_name") if isinstance(m, dict) else getattr(m, "english_name", None)
            if en == english_name:
                seq = (m.get("keywords") if isinstance(m, dict) else getattr(m, "keywords", None)) \
                      or (m.get("sample_courses") if isinstance(m, dict) else getattr(m, "sample_courses", None)) \
                      or []
                hint = ", ".join(seq[:6])
                break

        docs = self._retrieve_major_context(english_name, original_name, hint)
        context = "\n\n".join(d.page_content[:1000] for d in docs) if docs else ""

        if not (ChatPromptTemplate and self.llm and self.output_parser):
            # graceful fallback when prompt/llm/parser not wired
            return "High similarity to the student's profile based on rubric alignment."

        prompt = ChatPromptTemplate.from_template(
            "You are an Academic Advisor. Use the PDF context if relevant; otherwise, provide a general rationale. "
            "Write 2–3 concise sentences explaining why the following major fits the student.\n\n"
            "LANGUAGE: Respond ONLY in English. Keep the major name in English.\n\n"
            "Major (English name): {major_en}\n"
            "Student profile: {profile}\n"
            "PDF context (may be empty):\n{ctx}\n\n"
            "Return only the explanation in English."
        )
        chain = prompt | self.llm | self.output_parser
        try:
            why = chain.invoke(
                {"major_en": english_name, "profile": student_profile_text, "ctx": context}
            ).strip()
        except Exception:
            why = "High similarity to the student's profile based on rubric alignment."

        if docs:
            srcs = sorted({d.metadata.get("source") for d in docs if d.metadata.get("source")})
            if srcs:
                why += "\nSources: " + ", ".join(srcs)
        return why
