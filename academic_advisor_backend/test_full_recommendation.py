"""
Test full recommendation flow with Hebrew keywords
"""
import os
os.environ.setdefault("BAGRUT_JSON_PATH", "./state/extracted_bagrut.json")

from query.query import HybridRAG
from query.recommender import Recommender
import json

# Use REAL Bagrut from disk (user's actual file)
bagrut_path = "./state/extracted_bagrut.json"
if os.path.exists(bagrut_path):
    with open(bagrut_path, 'r', encoding='utf-8') as f:
        bagrut_json = json.load(f)
    print(f"✓ Loaded Bagrut from {bagrut_path}")
    print(f"  Subjects: {len(bagrut_json.get('subjects', []))}")
else:
    print(f"✗ Bagrut file not found: {bagrut_path}")
    exit(1)

# Load majors
majors_path = "./majors_data/majors_embeddings.json"
if os.path.exists(majors_path):
    with open(majors_path, 'r', encoding='utf-8') as f:
        majors = json.load(f)
    print(f"✓ Loaded {len(majors)} majors from {majors_path}")
else:
    print(f"✗ Majors file not found: {majors_path}")
    exit(1)

# Create RAG
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.0
)

rag = HybridRAG(ui_language="he", llm=llm)
rag.set_session_context(ui_language="he", bagrut_json=bagrut_json)

print("\n" + "=" * 70)
print("INITIAL PERSON VECTOR (Bagrut seed)")
print("=" * 70)
initial = rag.person.as_dict()
for k, v in sorted(initial.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"  {k}: {v:.1f}")

print("\n" + "=" * 70)
print("SIMULATING INTERVIEW: 'מתמטיקה אני אוהב, הסתברות וניתוח נתונים'")
print("=" * 70)

# Absorb answer
user_answer = "מתמטיקה אני אוהב, הסתברות, אלגוריתמים וניתוח נתונים"
last_question = "מה מעניין אותך?"

ok = rag.absorb_answer(user_text=user_answer, last_question=last_question)
print(f"\nAbsorb result: {ok}")

print("\n" + "=" * 70)
print("UPDATED PERSON VECTOR (after interview)")
print("=" * 70)
updated = rag.person.as_dict()
for k, v in sorted(updated.items(), key=lambda x: x[1], reverse=True)[:8]:
    delta = v - initial.get(k, 0)
    arrow = "↑" if delta > 1 else "↓" if delta < -1 else "="
    print(f"  {k}: {v:.1f} ({arrow} {abs(delta):.1f})")

# Get recommendations
print("\n" + "=" * 70)
print("GENERATING RECOMMENDATIONS")
print("=" * 70)

rag.recommender.majors = majors
rag.recommender.person = rag.person

results = rag.recommender.recommend_final(top_k=10)

print("\nTOP 10 RECOMMENDATIONS:")
for i, rec in enumerate(results, 1):
    name = rec.get("original_name", rec.get("english_name", "Unknown"))
    score = rec.get("match_percentage", 0)
    print(f"  [{i:2d}] {name[:60]:60s} {score:5.1f}%")

# Check if Computer Science is in top 5
cs_names = ["מדעי המחשב", "Computer Science", "מחשב"]
cs_found = False
cs_rank = None
for i, rec in enumerate(results[:5], 1):
    name = rec.get("original_name", "")
    if any(cs in name for cs in cs_names):
        cs_found = True
        cs_rank = i
        break

print("\n" + "=" * 70)
if cs_found:
    print(f"✓✓✓ SUCCESS! Computer Science recommended at rank #{cs_rank}")
else:
    print(f"✗✗✗ FAILED! Computer Science NOT in top 5")
    print("Top 5 majors:")
    for i, rec in enumerate(results[:5], 1):
        name = rec.get("original_name", "")
        print(f"  {i}. {name}")
print("=" * 70)
