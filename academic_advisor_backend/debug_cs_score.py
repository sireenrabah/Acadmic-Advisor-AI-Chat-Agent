#!/usr/bin/env python3
"""
Debug script to calculate Computer Science score manually
"""
import json
import numpy as np

# Load majors embeddings
with open("majors_data/majors_embeddings.json", "r", encoding="utf-8") as f:
    majors = json.load(f)

# Find Computer Science
cs = None
for major in majors:
    if "מדעי המחשב" in major.get("original_name", ""):
        cs = major
        break

if not cs:
    print("ERROR: Computer Science not found!")
    exit(1)

print(f"Found: {cs['original_name']}")
print(f"English name: {cs['english_name']}")
print()

# Student profile (from your test)
person_scores = {
    'english_proficiency': 98.6,
    'english_expression': 98.6,
    'hebrew_proficiency': 96.1,
    'hebrew_expression': 95.1,
    'logical_problem_solving': 94.2,
    'systems_thinking': 93.3,
    'communication_mediation': 89.8,
    'quantitative_reasoning': 89.1,
    'pattern_recognition': 88.1,
    'data_analysis': 86.5,
    'attention_to_detail': 84.5,
    'social_sensitivity': 82.7,
    'risk_decision': 74.5,
    'curiosity_interdisciplinary': 74.5,
    'ethical_awareness': 73.8,
    'theoretical_patience': 73.8,
    'strategic_thinking': 50.0,
    'cognitive_flexibility': 50.0,
    'creative_solutions': 50.0,
    'innovation_openness': 50.0,
    'opportunity_recognition': 50.0,
    'teamwork': 50.0,
    'psychological_interest': 50.0,
    'leadership': 50.0
}

# CS scores
cs_scores = cs['scores']

print("=== TECHNICAL CRITERIA COMPARISON ===")
technical = [
    'logical_problem_solving',
    'quantitative_reasoning', 
    'data_analysis',
    'pattern_recognition',
    'systems_thinking',
    'attention_to_detail',
    'theoretical_patience'
]

for crit in technical:
    person_val = person_scores[crit]
    cs_val = cs_scores[crit]
    diff = abs(person_val - cs_val)
    print(f"{crit:30s}: Person={person_val:5.1f}, CS={cs_val:5.1f}, Δ={diff:5.1f}")

print()
print("=== LANGUAGE CRITERIA COMPARISON ===")
language = [
    'hebrew_proficiency',
    'english_proficiency',
    'hebrew_expression',
    'english_expression'
]

for crit in language:
    person_val = person_scores[crit]
    cs_val = cs_scores[crit]
    diff = abs(person_val - cs_val)
    print(f"{crit:30s}: Person={person_val:5.1f}, CS={cs_val:5.1f}, Δ={diff:5.1f}")

print()
print("=== WEIGHTED RUBRIC CALCULATION ===")

# Calculate weighted rubric (like the code does)
criteria_keys = list(person_scores.keys())
weights = np.ones(len(criteria_keys))

high_priority = [
    'data_analysis', 'quantitative_reasoning', 'logical_problem_solving',
    'pattern_recognition', 'systems_thinking', 'theoretical_patience',
    'attention_to_detail'
]

low_priority = [
    'hebrew_proficiency', 'english_proficiency',
    'hebrew_expression', 'english_expression'
]

for i, key in enumerate(criteria_keys):
    if key in high_priority:
        weights[i] = 3.0
    elif key in low_priority:
        weights[i] = 0.3

# Build vectors
person_vec = np.array([person_scores[k] for k in criteria_keys])
cs_vec = np.array([cs_scores[k] for k in criteria_keys])

# Normalize to 0-1
person_norm = person_vec / 100.0
cs_norm = cs_vec / 100.0

# Weighted difference
weighted_diff = np.abs(person_norm - cs_norm) * weights
d = weighted_diff.sum() / weights.sum()

rubric_similarity = max(0.0, 1.0 - d)

print(f"Weighted mean difference: {d:.4f}")
print(f"Rubric similarity: {rubric_similarity:.4f} ({rubric_similarity*100:.1f}%)")

print()
print("=== COSINE SIMILARITY ===")
cosine = np.dot(person_vec, cs_vec) / (np.linalg.norm(person_vec) * np.linalg.norm(cs_vec))
print(f"Cosine: {cosine:.4f}")

print()
print("=== FINAL BLEND (weights: cosine=0.20, rubric=0.50, bagrut=0.30) ===")
# Assume bagrut_align = 0.85 (student has good math/english)
bagrut_align = 0.85
blend = 0.20 * cosine + 0.50 * rubric_similarity + 0.30 * bagrut_align
final_score = blend * 100.0

print(f"Blend (0-1): {blend:.4f}")
print(f"Final Score: {final_score:.1f}%")

print()
print("=== NOW CHECK INFORMATION SYSTEMS ===")
# Find Info Systems
info_sys = None
for major in majors:
    if "מערכות מידע" in major.get("original_name", ""):
        info_sys = major
        break

if info_sys:
    print(f"\nFound: {info_sys['original_name']}")
    is_vec = np.array([info_sys['scores'][k] for k in criteria_keys])
    is_norm = is_vec / 100.0
    
    # Weighted difference
    weighted_diff_is = np.abs(person_norm - is_norm) * weights
    d_is = weighted_diff_is.sum() / weights.sum()
    rubric_is = max(0.0, 1.0 - d_is)
    
    cosine_is = np.dot(person_vec, is_vec) / (np.linalg.norm(person_vec) * np.linalg.norm(is_vec))
    
    blend_is = 0.20 * cosine_is + 0.50 * rubric_is + 0.30 * bagrut_align
    final_is = blend_is * 100.0
    
    print(f"Rubric similarity: {rubric_is:.4f} ({rubric_is*100:.1f}%)")
    print(f"Cosine: {cosine_is:.4f}")
    print(f"Final Score: {final_is:.1f}%")
    
    print(f"\n*** DIFFERENCE: CS={final_score:.1f}%, IS={final_is:.1f}%, Δ={final_score-final_is:.1f}% ***")
