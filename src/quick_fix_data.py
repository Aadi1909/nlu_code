#!/usr/bin/env python3
"""
Quick fix script to improve model accuracy:
1. Analyze current training data issues
2. Augment data for intents with low F1 scores
3. Balance the dataset
4. Retrain the model
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict
import sys

def analyze_current_data():
    """Analyze training data distribution"""
    data_path = Path("../data/processed/train_augmented.json")
    
    # Try loading as regular JSON, if fails try line-by-line (JSONL)
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Try loading as JSONL (one JSON object per line)
        print("⚠️  File appears to be JSONL format, loading line by line...")
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    intents = [item['intent'] for item in data]
    intent_counts = Counter(intents)
    
    print("=" * 70)
    print("CURRENT TRAINING DATA ANALYSIS")
    print("=" * 70)
    print(f"\nTotal samples: {len(data)}")
    print(f"Total intents: {len(intent_counts)}")
    print(f"\nIntents with <20 samples (CRITICAL):")
    
    critical_intents = []
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1]):
        if count < 20:
            print(f"  ❌ {intent:30s}: {count:3d} samples")
            critical_intents.append((intent, count))
    
    print(f"\nIntents with 20-50 samples (NEEDS IMPROVEMENT):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1]):
        if 20 <= count < 50:
            print(f"  ⚠️  {intent:30s}: {count:3d} samples")
    
    print(f"\nIntents with 50+ samples (GOOD):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1]):
        if count >= 50:
            print(f"  ✅ {intent:30s}: {count:3d} samples")
    
    return data, intent_counts, critical_intents


def augment_text(text: str, variations: int = 3) -> List[str]:
    """Generate variations of a text"""
    augmented = [text]
    
    # Simple augmentation strategies
    strategies = [
        # Add filler words
        lambda t: random.choice(["please ", "kindly ", "मुझे ", ""]) + t,
        # Add question marks
        lambda t: t + random.choice(["?", "??", ""]),
        # Add polite endings
        lambda t: t + random.choice([" please", " पर", " है", ""]),
        # Change word order (simple)
        lambda t: " ".join(random.sample(t.split(), len(t.split()))) if len(t.split()) > 2 else t,
    ]
    
    for _ in range(variations):
        strategy = random.choice(strategies)
        try:
            aug_text = strategy(text)
            if aug_text not in augmented and len(aug_text) > 3:
                augmented.append(aug_text)
        except:
            pass
    
    return augmented[:variations + 1]


def balance_dataset(data: List[Dict], intent_counts: Counter, target_min: int = 50) -> List[Dict]:
    """Balance dataset by augmenting underrepresented intents"""
    
    print("\n" + "=" * 70)
    print("AUGMENTING DATA TO BALANCE DATASET")
    print("=" * 70)
    
    # Group by intent
    intent_to_examples = {}
    for item in data:
        intent = item['intent']
        if intent not in intent_to_examples:
            intent_to_examples[intent] = []
        intent_to_examples[intent].append(item)
    
    balanced_data = []
    
    for intent, examples in intent_to_examples.items():
        current_count = len(examples)
        
        if current_count >= target_min:
            # Already has enough samples
            balanced_data.extend(examples)
            print(f"✅ {intent:30s}: {current_count:3d} samples (sufficient)")
        else:
            # Need to augment
            needed = target_min - current_count
            print(f"⚙️  {intent:30s}: {current_count:3d} → augmenting by {needed}")
            
            # Add original examples
            balanced_data.extend(examples)
            
            # Generate augmented examples
            for _ in range(needed):
                # Pick random example to augment
                original = random.choice(examples)
                augmented_texts = augment_text(original['text'], variations=1)
                
                for aug_text in augmented_texts[1:]:  # Skip original
                    balanced_data.append({
                        'text': aug_text,
                        'intent': intent,
                        'label': original['label'],
                        'language': original.get('language', 'mixed')
                    })
                    if len([x for x in balanced_data if x['intent'] == intent]) >= target_min:
                        break
    
    return balanced_data


def save_balanced_data(balanced_data: List[Dict]):
    """Save balanced dataset"""
    output_path = Path("../data/processed/train_balanced.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved balanced dataset to: {output_path}")
    print(f"   Total samples: {len(balanced_data)}")
    
    # Show new distribution
    new_counts = Counter([item['intent'] for item in balanced_data])
    print(f"\n📊 New distribution:")
    for intent, count in sorted(new_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {intent:30s}: {count:3d} samples")


def main():
    print("🚀 Starting Model Improvement Process...\n")
    
    # Step 1: Analyze
    data, intent_counts, critical_intents = analyze_current_data()
    
    # Step 2: Balance
    balanced_data = balance_dataset(data, intent_counts, target_min=50)
    
    # Step 3: Save
    save_balanced_data(balanced_data)
    
    print("\n" + "=" * 70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the balanced dataset")
    print("2. Run training with: python train_intent_classifier_improved.py")
    print("   (This will use train_balanced.json automatically)")


if __name__ == "__main__":
    main()
