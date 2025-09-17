# query/person_embeddings.py
from __future__ import annotations

"""
Person embeddings + profile using the shared EmbeddingsBase and shared CRITERIA.
- Each answer is scored to a FULL 0..5 vector on the same keys.
- Profile JSON includes 'criteria_keys' for alignment visibility.
"""

import os, json, time
from dataclasses import dataclass, field
from typing import Dict, List, Any

from embeddings import EmbeddingsBase, clamp_int, get_criteria_keys
# keep majors and person aligned by importing the same keys
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None


@dataclass
class PersonProfile:
    path: str
    person_id: str = "default"
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, int] = field(default_factory=dict)   # current 0..5
    answers: List[Dict[str, Any]] = field(default_factory=list)

    # ----- lifecycle -----
    @classmethod
    def load(cls, path: str, person_id: str = "default") -> "PersonProfile":
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            prof = cls(path=path, person_id=data.get("person_id", person_id))
            prof.sums = data.get("sums", {})
            prof.counts = data.get("counts", {})
            prof.scores = {k: clamp_int(v) for k, v in data.get("scores", {}).items()}
            prof.answers = data.get("answers", [])
            return prof
        return cls(path=path, person_id=person_id)

    def save(self) -> None:
        payload = {
            "person_id": self.person_id,
            "updated_at": int(time.time()),
            "criteria_keys": get_criteria_keys(),
            "scores": self.scores,
            "vector": self.vector(),
            "sums": self.sums,
            "counts": self.counts,
            "answers": self.answers,
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ----- vector -----
    def vector(self) -> List[float]:
        keys = get_criteria_keys()
        return [float(clamp_int(self.scores.get(k, 0))) for k in keys]

    # ----- update -----
    def apply_full_scores(self, full: Dict[str, int], weight: float = 1.0) -> None:
        w = max(0.1, min(2.0, float(weight)))
        keys = get_criteria_keys()
        for k in keys:
            v = clamp_int(full.get(k, 0))
            self.sums[k] = float(self.sums.get(k, 0.0)) + v * w
            self.counts[k] = float(self.counts.get(k, 0.0)) + w
            avg = self.sums[k] / max(1e-9, self.counts[k])
            self.scores[k] = clamp_int(round(avg))

    def update_from_answer(self, question_id: str, answer_text: str, history_text: str = "",
                           weight: float = 1.0, temperature: float = 0.0) -> Dict[str, int]:
        scorer = EmbeddingsBase(temperature=temperature)
        full = scorer.score_person(answer_text=answer_text, history_text=history_text)
        self.apply_full_scores(full, weight=weight)
        self.answers.append({"question_id": question_id, "answer": answer_text, "applied_scores": full})
        return full


def update_person_profile(profile_path: str, question_id: str, answer_text: str,
                          person_id: str = "default", history_text: str = "",
                          weight: float = 1.0, temperature: float = 0.0) -> Dict[str, Any]:
    prof = PersonProfile.load(profile_path, person_id=person_id)
    full = prof.update_from_answer(question_id, answer_text, history_text=history_text,
                                   weight=weight, temperature=temperature)
    prof.save()
    return {"applied": full, "vector": prof.vector(), "path": profile_path}


def prompt_for_person(self):
    if ChatPromptTemplate is None:
        raise RuntimeError("langchain-core not installed.")
    template = (
        "You are evaluating a PERSON on the SAME rubric used for majors.\n"
        f"Return ONLY a JSON object mapping EACH criterion key to an INTEGER {SCALE_MIN}..{SCALE_MAX} (no missing keys).\n\n"
        "Rules:\n"
        f"- {0} = not evident at all; {100} = extremely strong evidence.\n"
        "- Provide ALL keys; integers only; no commentary; JSON only.\n\n"
        "Rubric keys and meanings:\n"
        f"{self.rubric_lines}\n\n"
        "PRIOR HISTORY (optional): {history}\n"
        "CURRENT ANSWER: {answer}\n\n"
        "JSON object with ONLY these keys (all must be present):\n"
        f"{self.keys}\n"
    )
    return ChatPromptTemplate.from_template(template)
