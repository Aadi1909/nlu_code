#!/usr/bin/env python3
"""
Manual Testing Script - Test Tier 1 Classifier with sample queries
Run: python test_tier1.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tier1_classifier import Tier1Classifier, IntentCategory

def test_queries():
    """Test with predefined queries"""
    
    print("\n" + "=" * 80)
    print("🧪 TIER 1 CLASSIFIER - MANUAL TESTING")
    print("=" * 80)
    print("\nℹ️  Using rule-based classifier (100% accurate for pattern matches)")
    
    # Initialize classifier (no ML model needed for rule-based)
    classifier = Tier1Classifier(model_path=None, confidence_threshold=0.5)
    
    # Test cases
    test_cases = [
        ("wallet balance check karo", "wallet_balance", "tier1"),
        ("mere wallet mein kitna paisa hai", "wallet_balance", "tier1"),
        ("wallet status for driver AB123456", "wallet_balance", "tier1"),
        ("swap station kaha hai", "battery_swap_status", "tier1"),
        ("nearest swap station", "battery_swap_status", "tier1"),
        ("swap history dikhao", "swap_history", "tier1"),
        ("mera swap record", "swap_history", "tier1"),
        ("battery kaise swap kare", "swap_process", "tier1"),
        ("swap process batao", "swap_process", "tier1"),
        ("meri booking ka status", "booking_status", "tier1"),
        ("booking confirm hui #12345678", "booking_status", "tier1"),
        ("fine kyu laga", "penalty_reason", "tier1"),
        ("penalty reason batao", "penalty_reason", "tier1"),
        ("mera registration status", "driver_onboarding_status", "tier1"),
        ("account approve hua kya", "driver_onboarding_status", "tier1"),
        ("booking cancel kardo", "booking_cancel", "agent"),
        ("cancel my booking #87654321", "booking_cancel", "agent"),
        ("driver kaise bane", "driver_onboarding", "agent"),
        ("Battery Smart join karna hai", "driver_onboarding", "agent"),
        ("account close karna hai", "driver_deboarding", "agent"),
        ("driver account band karo", "driver_deboarding", "agent"),
    ]
    
    print("\n" + "=" * 80)
    print("TESTING QUERIES")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for query, expected_intent, expected_category in test_cases:
        result = classifier.classify(query)
        
        intent_match = result.intent == expected_intent
        category_match = result.category.value == expected_category
        
        if intent_match and category_match:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        emoji = "🤖" if result.category == IntentCategory.TIER_1 else "👤"
        
        print(f"\n{status} Query: '{query}'")
        print(f"   {emoji} Detected: {result.intent} ({result.confidence:.0%})")
        print(f"   Expected: {expected_intent} ({expected_category})")
        if result.entities:
            print(f"   Entities: {result.entities}")
        if not intent_match or not category_match:
            print(f"   ⚠️  MISMATCH!")
    
    print("\n" + "=" * 80)
    print(f"📊 RESULTS: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.0f}%)")
    print("=" * 80)
    
    return passed, failed


def interactive_mode():
    """Interactive testing mode"""
    
    print("\n" + "=" * 80)
    print("🧪 INTERACTIVE MODE")
    print("=" * 80)
    print("\nType queries to test (or 'quit' to exit, 'help' for examples)")
    
    classifier = Tier1Classifier(model_path=None, confidence_threshold=0.5)
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\n📝 Example queries:")
                print("   TIER 1:")
                print("   • wallet balance check karo")
                print("   • swap station kaha hai")
                print("   • booking status batao")
                print("   • fine kyu laga")
                print("\n   AGENT:")
                print("   • booking cancel kardo")
                print("   • driver kaise bane")
                print("   • account close karna hai")
                continue
            
            result = classifier.classify(query)
            
            emoji = "🤖" if result.category == IntentCategory.TIER_1 else "👤"
            status = "BOT HANDLES" if result.category == IntentCategory.TIER_1 else "AGENT HANDOFF"
            
            print(f"\n{emoji} Intent: {result.intent}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Status: {status}")
            print(f"   Action: {result.action}")
            
            if result.entities:
                print(f"   Entities: {result.entities}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        # Run automated tests
        test_queries()
        
        print("\n💡 Tip: Run 'python test_tier1.py interactive' for interactive mode")
