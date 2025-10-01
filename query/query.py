# query/query.py
from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

# ---- Embeddings & Vector store (for PDF context in rationales) ----
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback

# ---- LLM (Gemini) ----
from langchain_google_genai import ChatGoogleGenerativeAI

# ---- Prompts (only for simple ask()) ----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---- Our modules ----
from generation.generate import QuestionGenerator
from query.recommender import Recommender
from query.person_embeddings import PersonSession, update_person_session
from embeddings import get_criteria_keys, SCALE_MIN, SCALE_MAX
CRITERIA_SET = set(get_criteria_keys())


# ========================== Data types ==========================

@dataclass
class MajorProfile:
    original_name: str          # as in the source (e.g., Hebrew)
    english_name: str           # English display name
    keywords: List[str]
    sample_courses: List[str]
    source: str = "majors.pdf"


# ========================== Orchestrator ==========================

class HybridRAG:
    """
    Minimal orchestrator:
      • Builds embeddings + Chroma for PDF context (used in recommender explanations).
      • Runs the deep, criterion-targeted interview loop.
      • Maintains a PersonSession (0..100 rubric vector) and logs after each answer.
      • Loads major rubric vectors and hands them to the Recommender.
    """

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "pdf_collection",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gemini-2.0-flash",
        k: int = 3,
    ):
        self.k = k
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # --- Embeddings (for vector DB / explanations only) ---
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except TypeError:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},
            )

        # --- Vector store (used by recommender for context snippets) ---
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )

        # --- LLM ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Put it in your environment or .env file.")
        self.llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0.2)
        self.output_parser = StrOutputParser()

        # --- Modules ---
        self.qgen = QuestionGenerator(llm=self.llm)
        self.recommender = Recommender(
            embeddings=self.embeddings,
            vectorstore=self.vectorstore,
            llm=self.llm,
            output_parser=self.output_parser,
        )

        # --- Majors memory (rubric-space vectors) ---
        self.majors: List[MajorProfile] = []
        self._major_vectors: List[Tuple[str, List[float]]] = []  # (english_name, rubric_vector 0..100)
        self._name_map: Dict[str, str] = {}  # english_name -> original_name

    # ===================== Ingestion / Upsert =====================

    def upsert_chunks(self, chunks: List) -> None:
        """Insert split LangChain Documents into Chroma (optional; for explanations)."""
        if not chunks:
            return
        self.vectorstore.add_documents(chunks)

    # ===================== Majors loading (rubric space) =====================

    def load_majors_rubric_vectors(self, path: str = "majors_embeddings.json") -> int:
        """
        Load majors + rubric vectors (0..100) and bind them to the recommender.
        Accepts flexible schemas:
          - [{"english_name": "...", "original_name": "...", "vector": [...]}, ...]
          - [{"english_name": "...", "scores": {"key": int, ...}}, ...]
        Unknown/missing criteria → 0. Vectors are clamped to 0..100 and aligned to get_criteria_keys().
        """
        if not os.path.exists(path):
            self.majors, self._major_vectors, self._name_map = [], [], {}
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[majors] Failed to read {path}: {e}")
            self.majors, self._major_vectors, self._name_map = [], [], {}
            return 0

        items = raw
        if isinstance(raw, dict):
            items = raw.get("items") or raw.get("majors") or []

        keys = get_criteria_keys()
        majors: List[MajorProfile] = []
        vectors: List[Tuple[str, List[float]]] = []
        name_map: Dict[str, str] = {}

        def _coerce_vec(obj: Dict[str, Any]) -> List[float]:
            # from explicit list
            if isinstance(obj.get("vector"), list):
                v = obj["vector"]
            # or from scores dict
            elif isinstance(obj.get("scores"), dict):
                v = [obj["scores"].get(k, 0) for k in keys]
            else:
                v = [0 for _ in keys]
            # clamp + align length
            v = [float(max(SCALE_MIN, min(SCALE_MAX, float(x)))) for x in (v[:len(keys)] + [0.0] * max(0, len(keys) - len(v)))]
            return v

        for it in items or []:
            en = str(it.get("english_name") or it.get("name_en") or it.get("name") or "").strip()
            if not en:
                # skip unnamed entries
                continue
            orig = str(it.get("original_name") or it.get("name_he") or en).strip()
            kws = [str(x).strip() for x in (it.get("keywords") or []) if str(x).strip()]
            courses = [str(x).strip() for x in (it.get("sample_courses") or []) if str(x).strip()]
            majors.append(MajorProfile(original_name=orig, english_name=en, keywords=kws, sample_courses=courses))
            vectors.append((en, _coerce_vec(it)))
            name_map[en] = orig

        self.majors, self._major_vectors, self._name_map = majors, vectors, name_map
        self.recommender.bind_majors(self.majors, self._major_vectors, self._name_map)
        return len(self.majors)

    # ========================= Simple Q&A (optional) ========================

    def ask(self, question: str) -> str:
        """Answer from general knowledge only (no retrieval). Language mirrors user."""
        system = (
            "You are an academic advisor assistant. Rely on your general knowledge (no external context).\n\n"
            "LANGUAGE POLICY:\n"
            "• Respond ONLY in the language of the user's question (English/Hebrew/Arabic).\n"
            "• Do NOT echo the question; output only the answer text."
        )
        tmpl = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question:\n{q}\n\nAnswer now, strictly following the LANGUAGE POLICY.")
        ])
        chain = tmpl | self.llm | self.output_parser
        return chain.invoke({"q": question}).strip()

    # ==================== Interview (criterion-targeted) ====================

    def run_interview(self, max_turns: int = 10):
        """
        Interactive CLI loop:
          • Asks one deep, criterion-targeted question at a time (QuestionGenerator).
          • Updates a 0..100 rubric vector from each answer (PersonSession).
          • Recommends top-3 majors by cosine similarity in rubric space.
        Logs after every answer to ./person_embeddings.jsonl and writes the latest vector to ./person_embeddings.json.
        """
        print("\n--- Academic Advisor Interview (criterion-targeted) ---")
        history: List[Tuple[str, str]] = []  # ("ai" or "user", text)
        asked_questions: List[str] = []
        turns = 0
        session = PersonSession()  # starts with all zeros (0..100)

        while True:
            majors_hint = ", ".join(m.english_name for m in self.majors[:20]) if self.majors else ""

            # Ask next question, steering by current scores so we fill the weakest/unknown facet first
            nxt = self.qgen.interview_turn(
                history=history,
                majors_hint=majors_hint,
                asked_questions=asked_questions,
                current_scores=session.scores,  # dict of key->0..100
            )

            if nxt.upper().startswith("RECOMMEND:"):
                block = self.recommender.final_block_from_person_vector(session.vector())
                print(block)
                return

            if not nxt.upper().startswith("QUESTION:"):
                # force a question if the model drifted
                nxt = self.qgen.interview_turn(
                    history=history,
                    majors_hint=majors_hint,
                    asked_questions=asked_questions,
                    force_question=True,
                    avoid_repeat_of=asked_questions[-1] if asked_questions else None,
                    current_scores=session.scores,
                )
                if not nxt.upper().startswith("QUESTION:"):
                    nxt = "QUESTION: You're given ambiguous requirements and a tight deadline. How would you break the problem down and choose a first step? Please add a brief example and (optionally) a 0–100 self-rating. [criterion=logical_problem_solving]"

            # Extract the plain question text and the criterion tag (if present)
            q_text = nxt.split(":", 1)[1].strip() if ":" in nxt else nxt.strip()
            mtag = re.search(r"\[criterion=([a-zA-Z0-9_]+)\]", q_text)
            criterion = mtag.group(1) if mtag else "unspecified"

            # Repeat-guard: avoid asking identical consecutive questions
            norm_q = re.sub(r"\s+", " ", q_text.lower()).strip()
            if asked_questions and norm_q == re.sub(r"\s+", " ", asked_questions[-1].lower()).strip():
                alt = self.qgen.interview_turn(
                    history=history,
                    majors_hint=majors_hint,
                    asked_questions=asked_questions,
                    avoid_repeat_of=q_text,
                    current_scores=session.scores,
                )
                if alt.upper().startswith("QUESTION:"):
                    q_text = alt.split(":", 1)[1].strip()

            print("\nAI:", q_text)
            user_ans = input("You: ").strip()
            if user_ans.lower() in ("stop", "exit", "quit"):
                print("Interview stopped.")
                break

            # Update session vector (0..100) and append logs
            if criterion in CRITERIA_SET:
                update_person_session(
                    session,
                    question_id=criterion,
                    answer_text=user_ans,
                    history_text="\n".join(t for _, t in history[-6:]),
                    weight=1.0,
                    temperature=0.0,
                )

            # update loop state
            history.append(("ai", q_text))
            history.append(("user", user_ans))
            asked_questions.append(q_text)
            turns += 1

            # stop condition
            if turns >= max_turns:
                print("\n[info] Max turns reached.")
                break

        # Final recommendation
        block = self.recommender.final_block_from_person_vector(session.vector())
        print(block)
