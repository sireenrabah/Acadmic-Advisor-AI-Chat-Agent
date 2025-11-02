# Quick debug of current Bagrut signals
import sys
sys.path.insert(0, r"c:\Users\marah\Documents\projects\fullstack+AI_projects\Acadmic-Advisor-AIChatAgent-final\academic_advisor_backend")

import json
from query.bagrut_features import bagrut_signals, normalize_bagrut

# Load current Bagrut
with open("state/extracted_bagrut.json", "r", encoding="utf-8") as f:
    bagrut = json.load(f)

print("="*80)
print("Current Bagrut State")
print("="*80)
print(f"Total subjects: {len(bagrut.get('subjects', []))}\n")

# Normalize and get signals
normalized = normalize_bagrut(bagrut)
print(f"Normalized subjects: {len(normalized.get('by_subject', {}))}")
print(f"Weighted average: {normalized.get('weighted_average')}\n")

print("="*80)
print("Bagrut Signals Calculation")
print("="*80)
signals = bagrut_signals(bagrut)

print(f"\n{'='*80}")
print("Summary")
print("="*80)
print(f"Total criteria with signals: {len(signals)}")
print(f"\nIf most criteria are LOW (<60), the problem is:")
print("  1. units=null in extracted_bagrut.json → low weight (0.5)")
print("  2. Subjects not mapped to SUBJECT2CRITERIA")
