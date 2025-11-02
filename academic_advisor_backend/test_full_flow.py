# Test: Simulate full flow with new criteria mapping
import sys, os
sys.path.insert(0, r"c:\Users\marah\Documents\projects\fullstack+AI_projects\Acadmic-Advisor-AIChatAgent-final\academic_advisor_backend")

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Set required env vars
os.environ["BAGRUT_JSON_PATH"] = os.path.join(os.getcwd(), "academic_advisor_backend", "state", "extracted_bagrut.json")

print("="*80)
print("Testing Full Flow with New Criteria Mapping")
print("="*80 + "\n")

# 1. Load embeddings
from embeddings.embeddings import get_criteria_keys
from embeddings.person_embeddings import PersonProfile

criteria_keys = get_criteria_keys()
print(f"✅ Loaded {len(criteria_keys)} criteria keys from embeddings.py\n")

# 2. Create mock Bagrut data (MUST match expected format!)
mock_bagrut = {
    "subjects": [
        {"subject": "Mathematics", "grade": 100, "units": 5},
        {"subject": "English", "grade": 95, "units": 5},
        {"subject": "Computer Science", "grade": 90, "units": 5},
        {"subject": "Physics", "grade": 85, "units": 5},
        {"subject": "Hebrew", "grade": 80, "units": 3}
    ]
}

# 3. Test PersonProfile seeding from Bagrut
from query.bagrut_features import bagrut_signals, seed_person_vector

print("📊 Seeding PersonProfile from mock Bagrut...")
profile = PersonProfile(criteria_keys=criteria_keys)

# Convert Bagrut to signals first
signals = bagrut_signals(mock_bagrut)
print(f"✅ Generated {len(signals)} signals from Bagrut")

# Now seed the profile
profile.vector = seed_person_vector(criteria_keys, signals)

print("\n🔍 PersonProfile after Bagrut seeding:")
scores = profile.as_dict()  # Returns {criterion_key: score} directly

# Show only NON-DEFAULT scores (changed from 50)
changed_criteria = {k: v for k, v in scores.items() if abs(v - 50.0) > 5.0}
if changed_criteria:
    print(f"✅ {len(changed_criteria)} criteria changed from default (50):")
    for crit, score in sorted(changed_criteria.items(), key=lambda x: -x[1]):
        print(f"   {crit}: {score:.1f}")
else:
    print("❌ ERROR: No criteria were updated! Bagrut seeding failed!")

# 4. Test recommendation scoring (if majors data exists)
print("\n" + "="*80)
print("🎯 Testing Recommendation Scoring")
print("="*80)

try:
    import json
    majors_path = os.path.join("majors_data", "majors_embeddings.json")
    extracted_path = os.path.join("majors_data", "extracted_majors.json")
    
    with open(majors_path, "r", encoding="utf-8") as f:
        majors_emb = json.load(f)
    with open(extracted_path, "r", encoding="utf-8") as f:
        majors_ext = json.load(f)
    
    print(f"✅ Loaded {len(majors_emb)} majors with embeddings\n")
    
    # Merge data
    majors = []
    ext_by_name = {m["english_name"]: m for m in majors_ext}
    for emb in majors_emb:
        name = emb.get("english_name") or emb.get("original_name")
        if name in ext_by_name:
            merged = {**ext_by_name[name], **emb}
            majors.append(merged)
    
    # Test scoring with a few relevant majors
    from query.recommender import Recommender
    
    recommender = Recommender(
        person=profile,  # FIX: parameter is 'person', not 'person_profile'
        majors=majors,
        ui_language="he"
    )
    
    print("🔍 Testing specific majors likely to match CS/Math student:\n")
    
    # Find Computer Science major
    cs_majors = [m for m in majors if "מדעי המחשב" in m.get("original_name", "")]
    math_majors = [m for m in majors if "מתמטיקה" in m.get("original_name", "") and "סטטיסטיקה" not in m.get("original_name", "")]
    
    test_majors = []
    if cs_majors:
        test_majors.append(("Computer Science", cs_majors[0]))
    if math_majors:
        test_majors.append(("Mathematics", math_majors[0]))
    
    # Also test an unrelated major
    unrelated = [m for m in majors if "אחיות" in m.get("original_name", "") or "ניהול" in m.get("original_name", "")]
    if unrelated:
        test_majors.append(("Unrelated (Nursing/Management)", unrelated[0]))
    
    for label, major in test_majors:
        # Calculate cosine similarity manually
        import numpy as np
        person_vec = np.array(profile.vector)  # FIX: use .vector not .to_vec()
        major_vec = major.get("vector", [])
        
        if len(person_vec) == len(major_vec):
            person_norm = np.linalg.norm(person_vec)
            major_norm = np.linalg.norm(major_vec)
            if person_norm > 0 and major_norm > 0:
                cos_sim = np.dot(person_vec, major_vec) / (person_norm * major_norm)
            else:
                cos_sim = 0.0
        else:
            cos_sim = 0.0
        
        print(f"📌 {label}: {major.get('original_name', 'Unknown')}")
        print(f"   Cosine similarity: {cos_sim:.3f} ({cos_sim*100:.1f}%)")
        
        # Show top matching criteria
        major_scores = major.get("scores", {})
        if major_scores:
            overlaps = []
            for crit_key in criteria_keys:
                person_score = scores.get(crit_key, 50.0)
                major_score = major_scores.get(crit_key, 50.0)
                # Only show if both are high (>70)
                if person_score > 70 and major_score > 70:
                    overlaps.append((crit_key, person_score, major_score))
            
            if overlaps:
                print(f"   Shared strengths ({len(overlaps)} criteria):")
                for crit, p_score, m_score in sorted(overlaps, key=lambda x: -(x[1]+x[2]))[:3]:
                    print(f"      - {crit}: person={p_score:.0f}, major={m_score:.0f}")
        print()
    
    print("="*80)
    print("✅ Test complete! Check if CS/Math have HIGH cosine similarity (>0.7)")
    print("   and unrelated majors have LOW similarity (<0.5)")
    
except FileNotFoundError as e:
    print(f"⚠️ Could not test recommendation scoring: {e}")
    print("   Run server first to load majors data")
except Exception as e:
    print(f"❌ Error during recommendation test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("🎯 SUMMARY")
print("="*80)
print("If you see:")
print("✅ Multiple criteria changed from 50 (especially quantitative_reasoning, logical_problem_solving)")
print("✅ CS/Math majors have cosine similarity > 0.70")
print("✅ Unrelated majors have cosine similarity < 0.50")
print("\nThen the fix is working! 🎉")
