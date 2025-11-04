"""
Test Hebrew keyword detection in absorb_answer
"""
import os
os.environ.setdefault("BAGRUT_JSON_PATH", "./state/extracted_bagrut.json")

from query.query import HybridRAG
from embeddings.person_embeddings import PersonProfile
from embeddings.embeddings import get_criteria_keys

# Mock minimal Bagrut
bagrut_json = {
    "subjects": [
        {"name": "Mathematics", "grade": 95, "units": 5},
        {"name": "English", "grade": 90, "units": 5}
    ]
}

# Create RAG with LLM
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.0
)

rag = HybridRAG(ui_language="he", llm=llm)
rag.set_session_context(ui_language="he", bagrut_json=bagrut_json)

print("=" * 60)
print("Initial person vector (Bagrut seed):")
initial = rag.person.as_dict()
for k, v in sorted(initial.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {k}: {v:.1f}")

print("\n" + "=" * 60)
print("Absorbing answer: 'מתמטיקה אני אוהב, הסתברות וניתוח נתונים'")
print("=" * 60)

# Simulate user answer
user_answer = "מתמטיקה אני אוהב, הסתברות וניתוח נתונים"
last_question = "מה מעניין אותך?"

ok = rag.absorb_answer(user_text=user_answer, last_question=last_question)

print("\n" + "=" * 60)
print(f"Absorb result: {ok}")
print("=" * 60)

print("\nUpdated person vector:")
updated = rag.person.as_dict()
for k, v in sorted(updated.items(), key=lambda x: x[1], reverse=True)[:10]:
    delta = v - initial.get(k, 0)
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"  {k}: {v:.1f} ({arrow} {abs(delta):.1f})")

print("\n" + "=" * 60)
print("Expected boosts:")
print("  quantitative_reasoning: Should be 95+ (was ~90)")
print("  logical_problem_solving: Should be 95+ (was ~94)")
print("  data_analysis: Should be 95+ (was ~86)")
print("=" * 60)

# Check critical criteria
critical_scores = {
    "quantitative_reasoning": updated["quantitative_reasoning"],
    "logical_problem_solving": updated["logical_problem_solving"],
    "data_analysis": updated["data_analysis"]
}

print("\nCritical scores after update:")
all_good = True
for criterion, score in critical_scores.items():
    status = "✓ PASS" if score >= 93 else "✗ FAIL"
    if score < 93:
        all_good = False
    print(f"  {criterion}: {score:.1f} {status}")

if all_good:
    print("\n✓✓✓ ALL TESTS PASSED! Hebrew keyword detection working!")
else:
    print("\n✗✗✗ TESTS FAILED! Scores still too low.")
