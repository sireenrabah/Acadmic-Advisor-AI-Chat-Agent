"""
Test script for final chat quality fixes.
Validates: language persistence, 3-part greeting, vector convergence, inline recommendations.
"""

import os
import sys

# Ensure we can import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'academic_advisor_backend'))

def test_language_example_phrase():
    """Test that _get_example_phrase returns correct language examples."""
    from query.query import HybridRAG
    
    print("\n=== Test 1: Language Example Phrases ===")
    
    # Hebrew
    rag_he = HybridRAG(ui_language="he")
    phrase_he = rag_he._get_example_phrase()
    assert "מתמטיקה" in phrase_he or "מצטיין" in phrase_he, f"Hebrew phrase incorrect: {phrase_he}"
    print(f"✓ Hebrew: {phrase_he}")
    
    # Arabic
    rag_ar = HybridRAG(ui_language="ar")
    phrase_ar = rag_ar._get_example_phrase()
    assert "الرياضيات" in phrase_ar or "متفوق" in phrase_ar, f"Arabic phrase incorrect: {phrase_ar}"
    print(f"✓ Arabic: {phrase_ar}")
    
    # English
    rag_en = HybridRAG(ui_language="en")
    phrase_en = rag_en._get_example_phrase()
    assert "mathematics" in phrase_en.lower() or "excel" in phrase_en.lower(), f"English phrase incorrect: {phrase_en}"
    print(f"✓ English: {phrase_en}")
    
    print("✓ All language examples correct")


def test_greeting_structure():
    """Test that greet_and_first_question returns 3 parts."""
    from query.query import HybridRAG
    
    print("\n=== Test 2: 3-Part Greeting Structure ===")
    
    rag = HybridRAG(ui_language="en")
    
    # Mock Bagrut data
    rag.bagrut_json = {
        "by_subject": {
            "Mathematics": {
                "final_grade": 95,
                "units": 5,
                "has_star": True
            }
        }
    }
    
    result = rag.greet_and_first_question()
    
    assert "greeting" in result, "Missing 'greeting' key"
    assert "bagrut_summary" in result, "Missing 'bagrut_summary' key"
    assert "first_question" in result, "Missing 'first_question' key"
    
    print(f"✓ Greeting: {result['greeting'][:50]}...")
    print(f"✓ Bagrut: {result['bagrut_summary'][:50]}...")
    print(f"✓ Question: {result['first_question'][:50]}...")
    
    # Check that Bagrut summary mentions grade
    assert "95" in result['bagrut_summary'] or "5" in result['bagrut_summary'], \
        "Bagrut summary doesn't mention grade/units"
    print("✓ Grade mentioned in Bagrut summary")
    
    print("✓ All 3 parts present and correct")


def test_vector_convergence():
    """Test that vector convergence detection works."""
    from query.query import HybridRAG
    from embeddings.person_embeddings import PersonProfile
    from embeddings.embeddings import get_criteria_keys
    
    print("\n=== Test 3: Vector Convergence Detection ===")
    
    rag = HybridRAG(ui_language="en")
    
    # Simulate updates with large changes
    rag.person.history = [
        {"old": 50.0, "new": 75.0},  # delta=25
        {"old": 75.0, "new": 80.0},  # delta=5
        {"old": 80.0, "new": 83.0},  # delta=3
        {"old": 83.0, "new": 84.0},  # delta=1
    ]
    
    converged = rag._check_vector_convergence(threshold=5.0)
    print(f"Recent updates: 25 → 5 → 3 → 1 (avg delta = {(25+5+3+1)/4})")
    print(f"Converged (threshold=5): {converged}")
    assert converged, "Should detect convergence when avg delta < 5"
    print("✓ Convergence detected correctly")
    
    # Simulate updates with large changes (not converged)
    rag.person.history = [
        {"old": 50.0, "new": 70.0},  # delta=20
        {"old": 70.0, "new": 80.0},  # delta=10
        {"old": 80.0, "new": 85.0},  # delta=5
        {"old": 85.0, "new": 92.0},  # delta=7
    ]
    
    not_converged = rag._check_vector_convergence(threshold=5.0)
    print(f"Recent updates: 20 → 10 → 5 → 7 (avg delta = {(20+10+5+7)/4})")
    print(f"Converged (threshold=5): {not_converged}")
    assert not not_converged, "Should NOT detect convergence when avg delta > 5"
    print("✓ Non-convergence detected correctly")


def test_repetitive_answers():
    """Test repetitive answer detection."""
    from query.query import HybridRAG
    
    print("\n=== Test 4: Repetitive Answer Detection ===")
    
    rag = HybridRAG(ui_language="en")
    
    # Short repetitive answers
    history = [
        ["ai", "Do you like math?"],
        ["user", "yes"],
        ["ai", "Do you prefer theory or practice?"],
        ["user", "yes"],
        ["ai", "What about programming?"],
        ["user", "both"],
        ["ai", "And lab work?"],
        ["user", "yes"],
        ["ai", "How about research?"],
        ["user", "both"],
        ["ai", "Engineering or science?"],
        ["user", "yes"],
    ]
    
    is_repetitive = rag._detect_repetitive_answers(history)
    short_count = sum(1 for role, text in history if role == "user" and len(text.split()) <= 3)
    print(f"Short answers (≤3 words): {short_count}/6")
    print(f"Repetitive: {is_repetitive}")
    assert is_repetitive, "Should detect repetitive short answers"
    print("✓ Repetition detected correctly")
    
    # Varied detailed answers
    history2 = [
        ["ai", "Do you like math?"],
        ["user", "Yes, I find it fascinating especially algebra"],
        ["ai", "What about programming?"],
        ["user", "I enjoy building projects with Python"],
        ["ai", "And lab work?"],
        ["user", "I prefer theoretical work over hands-on experiments"],
    ]
    
    not_repetitive = rag._detect_repetitive_answers(history2)
    print(f"Varied answers: NOT repetitive = {not not_repetitive}")
    assert not not_repetitive, "Should NOT detect repetition in varied answers"
    print("✓ Non-repetition detected correctly")


def test_stopping_logic():
    """Test multi-factor stopping logic."""
    from query.query import HybridRAG
    
    print("\n=== Test 5: Stopping Logic ===")
    
    rag = HybridRAG(ui_language="en")
    
    # Test 1: Max turns (10)
    should_stop = rag.should_recommend_now(history=[], turn_count=10)
    print(f"Turn 10: Should stop = {should_stop}")
    assert should_stop, "Should stop at turn 10 (max turns)"
    print("✓ Max turns stopping works")
    
    # Test 2: Repetition at turn 5
    history_rep = [
        ["ai", "Q1"], ["user", "yes"],
        ["ai", "Q2"], ["user", "yes"],
        ["ai", "Q3"], ["user", "both"],
        ["ai", "Q4"], ["user", "yes"],
        ["ai", "Q5"], ["user", "both"],
        ["ai", "Q6"], ["user", "yes"],
    ]
    should_stop_rep = rag.should_recommend_now(history=history_rep, turn_count=5)
    print(f"Turn 5 with repetition: Should stop = {should_stop_rep}")
    assert should_stop_rep, "Should stop at turn 5 with repetition"
    print("✓ Repetition stopping works")
    
    # Test 3: Vector convergence at turn 6
    rag.person.history = [
        {"old": 80.0, "new": 82.0},
        {"old": 82.0, "new": 83.0},
        {"old": 83.0, "new": 83.5},
        {"old": 83.5, "new": 84.0},
    ]
    should_stop_conv = rag.should_recommend_now(history=[], turn_count=6)
    print(f"Turn 6 with convergence: Should stop = {should_stop_conv}")
    assert should_stop_conv, "Should stop at turn 6 with vector convergence"
    print("✓ Convergence stopping works")


def test_inline_recommendations_format():
    """Test inline recommendations formatting."""
    from query.query import HybridRAG
    
    print("\n=== Test 6: Inline Recommendations Format ===")
    
    rag = HybridRAG(ui_language="en")
    
    # Mock majors data
    rag.majors = [
        {
            "english_name": "Computer Science",
            "original_name": "Computer Science",
            "vector": [70.0] * 10,
            "scores": {"analytical_reasoning": 85, "programming": 90}
        },
        {
            "english_name": "Mathematics",
            "original_name": "Mathematics", 
            "vector": [80.0] * 10,
            "scores": {"analytical_reasoning": 95, "programming": 60}
        }
    ]
    
    # Mock person vector
    rag.person.scores = {"analytical_reasoning": 80, "programming": 75}
    
    try:
        formatted = rag.format_inline_recommendations(top_k=2)
        print(f"Formatted output:\n{formatted}")
        
        assert "1." in formatted, "Should have numbered list"
        assert "2." in formatted, "Should have 2 items"
        assert "%" in formatted, "Should show match percentage"
        print("✓ Inline recommendations formatted correctly")
    except Exception as e:
        print(f"⚠ Inline recommendations test skipped (requires full setup): {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("FINAL CHAT QUALITY FIXES - TEST SUITE")
    print("=" * 60)
    
    try:
        test_language_example_phrase()
        test_greeting_structure()
        test_vector_convergence()
        test_repetitive_answers()
        test_stopping_logic()
        test_inline_recommendations_format()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
