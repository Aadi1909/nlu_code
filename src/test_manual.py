#!/usr/bin/env python3
"""
Interactive Manual Testing for Tier 1 Classifier

Run: python test_manual.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from tier1_classifier import Tier1Classifier, IntentCategory

def main():
    print("\n" + "=" * 70)
    print("🧪 BATTERY SMART - TIER 1 CLASSIFIER MANUAL TESTING")
    print("=" * 70)
    
    # Try to load ML model if available
    model_paths = [
        "/mnt/sagemaker-nvme/intent_models_full",
        "/mnt/sagemaker-nvme/intent_models",
        "../models/intent_classifier",
        "models/intent_classifier"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            print(f"✅ Found ML model at: {path}")
            break
    
    if not model_path:
        print("⚠️  No ML model found - using rule-based classifier only")
    
    classifier = Tier1Classifier(model_path=model_path, confidence_threshold=0.5)
    
    print("\n📋 Supported Intents:")
    print("-" * 50)
    print("\n🤖 TIER 1 (Bot Handles):")
    for intent, config in classifier.INTENT_CONFIG.items():
        if config['category'] == IntentCategory.TIER_1:
            print(f"   • {intent}: {config['description']}")
    
    print("\n👤 AGENT HANDOFF:")
    for intent, config in classifier.INTENT_CONFIG.items():
        if config['category'] == IntentCategory.AGENT:
            print(f"   • {intent}: {config['description']}")
    
    print("\n" + "=" * 70)
    print("Type a query to test (or 'quit' to exit, 'examples' for sample queries)")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'examples':
                print_examples()
                continue
            
            if query.lower() == 'help':
                print_help()
                continue
            
            # Classify
            result = classifier.classify(query)
            
            # Display result
            print("\n" + "-" * 50)
            
            if result.category == IntentCategory.TIER_1:
                emoji = "🤖"
                status = "BOT CAN HANDLE"
            else:
                emoji = "👤"
                status = "AGENT HANDOFF"
            
            print(f"{emoji} Intent: {result.intent}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Method: {result.method}")
            print(f"   Status: {status}")
            print(f"   Action: {result.action}")
            
            if result.entities:
                print(f"   Entities: {result.entities}")
            
            if result.top_predictions and len(result.top_predictions) > 1:
                print(f"   Other predictions:")
                for pred in result.top_predictions[1:3]:
                    print(f"      - {pred['intent']}: {pred['confidence']:.1%}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_examples():
    """Print example queries"""
    print("\n📝 Example Queries:")
    print("-" * 50)
    
    examples = {
        "🤖 TIER 1 (Bot Handles)": [
            "wallet balance check karo",
            "mere wallet mein kitna paisa hai",
            "swap station kaha hai",
            "nearest swap station",
            "swap history dikhao",
            "battery kaise swap kare",
            "meri booking ka status",
            "fine kyu laga",
            "mera registration status",
        ],
        "👤 AGENT HANDOFF": [
            "booking cancel kardo",
            "cancel my booking #12345678",
            "driver kaise bane",
            "Battery Smart join karna hai",
            "account close karna hai",
        ]
    }
    
    for category, queries in examples.items():
        print(f"\n{category}:")
        for q in queries:
            print(f"   • {q}")


def print_help():
    """Print help"""
    print("\n📖 Commands:")
    print("   • Type any query to test")
    print("   • 'examples' - Show sample queries")
    print("   • 'help' - Show this help")
    print("   • 'quit' or 'q' - Exit")


if __name__ == "__main__":
    main()
