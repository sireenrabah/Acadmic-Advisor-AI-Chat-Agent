# Test script to verify SUBJECT2CRITERIA matches CRITERIA keys
import sys
sys.path.insert(0, r"c:\Users\marah\Documents\projects\fullstack+AI_projects\Acadmic-Advisor-AIChatAgent-final\academic_advisor_backend")

from embeddings.embeddings import get_criteria_keys
from query.bagrut_features import SUBJECT2CRITERIA

# Get all criteria from embeddings.py
criteria_set = set(get_criteria_keys())
print(f"✅ Total CRITERIA keys in embeddings.py: {len(criteria_set)}")
print(f"Keys: {sorted(criteria_set)}\n")

# Check all SUBJECT2CRITERIA mappings
all_bagrut_criteria = set()
errors = []

for subject, crits in SUBJECT2CRITERIA.items():
    all_bagrut_criteria.update(crits)
    missing = [c for c in crits if c not in criteria_set]
    if missing:
        errors.append(f"❌ {subject}: Missing criteria - {missing}")
    else:
        print(f"✅ {subject}: All {len(crits)} criteria exist")

print("\n" + "="*80)
if errors:
    print("🔴 ERRORS FOUND:\n")
    for err in errors:
        print(err)
    print(f"\n❌ Total unique criteria used in SUBJECT2CRITERIA: {len(all_bagrut_criteria)}")
    print(f"❌ Criteria in SUBJECT2CRITERIA but NOT in CRITERIA: {all_bagrut_criteria - criteria_set}")
else:
    print("🎉 SUCCESS! All SUBJECT2CRITERIA mappings are valid!")
    print(f"✅ Total unique criteria used: {len(all_bagrut_criteria)}/{len(criteria_set)}")
    print(f"✅ Unused criteria (OK): {len(criteria_set - all_bagrut_criteria)}")
    print(f"   {sorted(criteria_set - all_bagrut_criteria)}")
