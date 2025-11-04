"""
Test script to verify chat improvements work correctly.
Run this to test the greeting flow and language persistence.
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "academic_advisor_backend"
sys.path.insert(0, str(backend_dir))

os.environ["BAGRUT_JSON_PATH"] = str(backend_dir / "state" / "extracted_bagrut.json")

from query.query import HybridRAG
from dotenv import load_dotenv

load_dotenv()

def test_greeting_and_first_question():
    """Test that greeting is separate from first question."""
    print("\n" + "="*60)
    print("TEST 1: Greeting + First Question (English)")
    print("="*60)
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
    except Exception as e:
        print(f"⚠️  LLM not available: {e}")
        llm = None
    
    rag = HybridRAG(ui_language="en", llm=llm)
    
    # Test with no Bagrut
    result = rag.greet_and_first_question()
    print(f"\n📝 WITHOUT BAGRUT:")
    print(f"  Greeting: {result['greeting']}")
    print(f"  Question: {result['first_question']}")
    
    # Test with Bagrut (if file exists)
    bagrut_path = backend_dir / "state" / "extracted_bagrut.json"
    if bagrut_path.exists():
        from query.bagrut_features import load_bagrut, bagrut_signals
        rag.bagrut_json = load_bagrut(str(bagrut_path))
        rag.signals = bagrut_signals(rag.bagrut_json)
        
        result = rag.greet_and_first_question()
        print(f"\n📝 WITH BAGRUT:")
        print(f"  Greeting: {result['greeting']}")
        print(f"  Question: {result['first_question']}")
    else:
        print(f"\n⚠️  No Bagrut file found at {bagrut_path}")


def test_language_persistence():
    """Test that Hebrew/Arabic are maintained in prompts."""
    print("\n" + "="*60)
    print("TEST 2: Language Persistence (Hebrew)")
    print("="*60)
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
    except Exception as e:
        print(f"⚠️  LLM not available: {e}")
        return
    
    rag = HybridRAG(ui_language="he", llm=llm)
    
    # Check example phrase
    phrase = rag._get_example_phrase()
    print(f"\n📝 Example phrase in Hebrew: {phrase}")
    
    # Test greeting
    result = rag.greet_and_first_question()
    print(f"\n📝 Greeting (should be Hebrew): {result['greeting']}")
    print(f"📝 Question (should be Hebrew): {result['first_question']}")
    
    # Verify it contains Hebrew characters
    has_hebrew = any('\u0590' <= c <= '\u05FF' for c in result['greeting'] + result['first_question'])
    if has_hebrew:
        print("✅ Hebrew detected in output!")
    else:
        print("❌ WARNING: No Hebrew detected - language may not be persisting")


def test_conversation_flow():
    """Test that conversation builds naturally."""
    print("\n" + "="*60)
    print("TEST 3: Conversation Flow")
    print("="*60)
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
    except Exception as e:
        print(f"⚠️  LLM not available: {e}")
        return
    
    rag = HybridRAG(ui_language="en", llm=llm)
    
    # Simulate conversation
    history = []
    asked = []
    
    # First question
    result = rag.greet_and_first_question()
    q1 = result['first_question']
    print(f"\n📝 Q1: {q1}")
    asked.append(q1)
    
    # Simulate short answer
    rag.last_answer_text = "Yes"
    history.append(["user", "Yes"])
    
    # Next question should probe for concrete examples
    q2 = rag.ask_next_question(history=history, asked_questions=asked)
    print(f"📝 Q2 (after short answer): {q2}")
    asked.append(q2)
    
    # Simulate detailed answer
    rag.last_answer_text = "I really enjoy solving complex problems and building systems. I like when there's a clear logical structure."
    history.append(["user", rag.last_answer_text])
    
    # Next question should narrow down
    q3 = rag.ask_next_question(history=history, asked_questions=asked)
    print(f"📝 Q3 (after detailed answer): {q3}")
    
    print("\n✅ Conversation flow test complete!")


def test_person_embedding_update():
    """Test that person embeddings update correctly."""
    print("\n" + "="*60)
    print("TEST 4: Person Embedding Updates")
    print("="*60)
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
    except Exception as e:
        print(f"⚠️  LLM not available: {e}")
        llm = None
    
    rag = HybridRAG(ui_language="en", llm=llm)
    
    # Get initial scores
    initial_scores = rag.person.as_dict()
    print(f"\n📊 Initial scores sample:")
    for k, v in list(initial_scores.items())[:5]:
        print(f"  {k}: {v:.1f}")
    
    # Absorb an answer
    success = rag.absorb_answer(
        user_text="I love programming and building software systems",
        last_question="What interests you most?"
    )
    
    if success:
        updated_scores = rag.person.as_dict()
        print(f"\n📊 Updated scores sample:")
        for k, v in list(updated_scores.items())[:5]:
            change = v - initial_scores[k]
            print(f"  {k}: {v:.1f} (change: {change:+.1f})")
        print("\n✅ Person embedding updated successfully!")
    else:
        print("\n❌ Failed to update person embedding")


if __name__ == "__main__":
    print("\n🚀 Testing Chat Improvements")
    print("="*60)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  WARNING: GOOGLE_API_KEY not set in environment")
        print("   Some tests will be skipped\n")
    
    test_greeting_and_first_question()
    test_language_persistence()
    test_conversation_flow()
    test_person_embedding_update()
    
    print("\n" + "="*60)
    print("✅ All tests complete!")
    print("="*60)
