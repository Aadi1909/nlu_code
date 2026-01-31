#!/usr/bin/env python3
"""
Process the raw training_data.json and prepare it for training.
This file contains 6000+ diverse examples that we should be using!
"""

import json
from pathlib import Path
from collections import Counter
import random

def load_raw_training_data():
    """Load the raw training data JSON"""
    data_path = Path("../data/raw/training_data.json")
    
    print(f"Loading raw training data from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} training examples!")
    return data

def analyze_data(data):
    """Analyze the training data distribution"""
    intent_counts = Counter([item['intent'] for item in data])
    language_counts = Counter([item.get('language', 'unknown') for item in data])
    
    print("\n" + "=" * 70)
    print("TRAINING DATA ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Total samples: {len(data)}")
    print(f"📊 Total intents: {len(intent_counts)}")
    print(f"📊 Languages: {dict(language_counts)}")
    
    print(f"\n🎯 Intent Distribution (top 20):")
    for intent, count in intent_counts.most_common(20):
        print(f"   {intent:35s}: {count:4d} samples")
    
    print(f"\n⚠️  Intents with <50 samples:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1]):
        if count < 50:
            print(f"   {intent:35s}: {count:4d} samples")
    
    return intent_counts

def create_label_mapping(data):
    """Create label mapping from the data"""
    intents = sorted(set(item['intent'] for item in data))
    
    label2id = {intent: idx for idx, intent in enumerate(intents)}
    id2label = {idx: intent for intent, idx in label2id.items()}
    
    return {"label2id": label2id, "id2label": id2label}

def prepare_train_val_test_split(data, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test sets"""
    
    # Group by intent for stratified split
    intent_to_examples = {}
    for item in data:
        intent = item['intent']
        if intent not in intent_to_examples:
            intent_to_examples[intent] = []
        intent_to_examples[intent].append(item)
    
    train_data = []
    val_data = []
    test_data = []
    
    for intent, examples in intent_to_examples.items():
        random.shuffle(examples)
        
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data.extend(examples[:train_end])
        val_data.extend(examples[train_end:val_end])
        test_data.extend(examples[val_end:])
    
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def save_processed_data(train_data, val_data, test_data, label_mapping):
    """Save processed data"""
    output_dir = Path("../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add labels to data
    def add_labels(data, label_mapping):
        for item in data:
            item['label'] = label_mapping['label2id'][item['intent']]
        return data
    
    train_data = add_labels(train_data, label_mapping)
    val_data = add_labels(val_data, label_mapping)
    test_data = add_labels(test_data, label_mapping)
    
    # Save as JSON (proper format)
    with open(output_dir / "train_full.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "val_full.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "test_full.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Save label mapping
    with open(output_dir / "label_mapping_full.json", 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved processed data:")
    print(f"   Train: {len(train_data)} samples → {output_dir / 'train_full.json'}")
    print(f"   Val:   {len(val_data)} samples → {output_dir / 'val_full.json'}")
    print(f"   Test:  {len(test_data)} samples → {output_dir / 'test_full.json'}")
    print(f"   Labels: {len(label_mapping['label2id'])} intents → {output_dir / 'label_mapping_full.json'}")

def main():
    print("🚀 Processing Raw Training Data...\n")
    
    # Load data
    data = load_raw_training_data()
    
    # Analyze
    intent_counts = analyze_data(data)
    
    # Create label mapping
    label_mapping = create_label_mapping(data)
    print(f"\n📋 Created label mapping with {len(label_mapping['label2id'])} intents")
    
    # Split data
    print("\n⚙️  Splitting data (80% train, 10% val, 10% test)...")
    train_data, val_data, test_data = prepare_train_val_test_split(data)
    
    # Save
    save_processed_data(train_data, val_data, test_data, label_mapping)
    
    print("\n" + "=" * 70)
    print("✅ DATA PROCESSING COMPLETE!")
    print("=" * 70)
    print("\nNext step: Train with full data")
    print("  python train_with_full_data.py")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
