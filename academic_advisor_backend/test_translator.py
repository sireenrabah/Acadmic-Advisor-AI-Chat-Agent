# Test UITranslator
import sys, os
sys.path.insert(0, r"c:\Users\marah\Documents\projects\fullstack+AI_projects\Acadmic-Advisor-AIChatAgent-final\academic_advisor_backend")

# Load .env
from dotenv import load_dotenv
load_dotenv()

from query.bagrut_features import _ui_tr, to_canonical_subject_en

tr = _ui_tr()
print(f"UITranslator type: {type(tr)}")
print(f"Has llm: {hasattr(tr, 'llm')}")

test_subjects = [
    "מתמטיקה",
    "אנגלית",
    "פיזיקה",
    "היסטוריה לערבים",
    "אלקטרוניקה ומחשבים"
]

print("\n" + "="*80)
print("Translation Tests")
print("="*80)
for subj in test_subjects:
    result = to_canonical_subject_en(subj)
    print(f"{subj:30s} → {result}")
