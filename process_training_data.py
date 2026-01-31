#!/usr/bin/env python3
"""
Script to process and split the enhanced training data into train, val, and test sets.
This creates the processed data files needed for model training.
"""

import json
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> list:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data: list, filepath: str):
    """Save data as JSON Lines format (one JSON object per line)."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: list, filepath: str):
    """Save data as regular JSON array."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_label_mapping(data: list) -> dict:
    """Create label to ID mapping from the data."""
    intents = sorted(set(item['intent'] for item in data))
    label2id = {label: idx for idx, label in enumerate(intents)}
    id2label = {idx: label for label, idx in label2id.items()}
    return {"label2id": label2id, "id2label": id2label}


def add_labels_to_data(data: list, label_mapping: dict) -> list:
    """Add numeric labels to each data item."""
    for item in data:
        item['label'] = label_mapping['label2id'][item['intent']]
    return data


def stratified_split(data: list, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """
    Perform stratified split of data into train, validation, and test sets.
    Handles cases where some classes have very few samples.
    """
    # Get intent distribution
    intent_counts = Counter(item['intent'] for item in data)
    
    # Separate data into groups based on sample count
    # For intents with very few samples, we'll ensure at least 1 sample in each split
    low_sample_threshold = 10
    
    low_sample_data = []
    high_sample_data = []
    
    for item in data:
        if intent_counts[item['intent']] < low_sample_threshold:
            low_sample_data.append(item)
        else:
            high_sample_data.append(item)
    
    # Split high sample data normally
    if high_sample_data:
        intents = [item['intent'] for item in high_sample_data]
        train_val, test = train_test_split(
            high_sample_data, 
            test_size=test_size, 
            stratify=intents, 
            random_state=random_state
        )
        
        train_val_intents = [item['intent'] for item in train_val]
        adjusted_val_size = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val_size,
            stratify=train_val_intents,
            random_state=random_state
        )
    else:
        train, val, test = [], [], []
    
    # Handle low sample data - distribute proportionally
    if low_sample_data:
        random.seed(random_state)
        random.shuffle(low_sample_data)
        
        # Group by intent
        low_sample_by_intent = {}
        for item in low_sample_data:
            intent = item['intent']
            if intent not in low_sample_by_intent:
                low_sample_by_intent[intent] = []
            low_sample_by_intent[intent].append(item)
        
        # Distribute each intent's samples
        for intent, items in low_sample_by_intent.items():
            n = len(items)
            if n >= 3:
                # At least 1 for test, 1 for val, rest for train
                test_n = max(1, int(n * test_size))
                val_n = max(1, int(n * val_size))
                train_n = n - test_n - val_n
                
                test.extend(items[:test_n])
                val.extend(items[test_n:test_n+val_n])
                train.extend(items[test_n+val_n:])
            elif n == 2:
                # 1 for train, 1 for val
                train.extend(items[:1])
                val.extend(items[1:])
            else:
                # All for train
                train.extend(items)
    
    # Shuffle all splits
    random.seed(random_state)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def upsample_train_set(train_data: list, min_samples_per_intent: int = 50, random_state: int = 42) -> list:
    """
    Upsample low-frequency intents in the training set to improve class balance.
    This duplicates existing samples (with a small metadata tag) to reach a minimum count.
    """
    rng = random.Random(random_state)
    intent_groups = {}
    for item in train_data:
        intent_groups.setdefault(item["intent"], []).append(item)

    balanced_train = list(train_data)

    for intent, items in intent_groups.items():
        current_count = len(items)
        if current_count >= min_samples_per_intent:
            continue

        needed = min_samples_per_intent - current_count
        for _ in range(needed):
            source_item = rng.choice(items)
            cloned = dict(source_item)
            cloned["metadata"] = dict(source_item.get("metadata", {}))
            cloned["metadata"].update({
                "augmented": True,
                "augmentation": "upsample"
            })
            balanced_train.append(cloned)

    rng.shuffle(balanced_train)
    return balanced_train


def validate_data(data: list) -> tuple:
    """Validate the data and return any issues found."""
    issues = []
    valid_data = []
    
    for idx, item in enumerate(data):
        # Check required fields
        if 'text' not in item or not item['text']:
            issues.append(f"Item {idx}: Missing or empty 'text' field")
            continue
        if 'intent' not in item or not item['intent']:
            issues.append(f"Item {idx}: Missing or empty 'intent' field")
            continue
        
        # Clean the text
        item['text'] = item['text'].strip()
        
        # Ensure entities is a list
        if 'entities' not in item:
            item['entities'] = []
        
        # Clean entity values
        if item.get('entities'):
            cleaned_entities = []
            for entity in item['entities']:
                if entity.get('value') is not None and entity.get('start') is not None:
                    cleaned_entities.append(entity)
            item['entities'] = cleaned_entities
        
        valid_data.append(item)
    
    return valid_data, issues


def main():
    # Paths - Use training_data_cleaned.json and filter to 19 intents
    raw_data_path = Path("data/raw/training_data_cleaned.json")
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Original 19 intents to keep
    VALID_INTENTS = {
        "driver_deboarding", "driver_onboarding", "driver_onboarding_status",
        "driver_wants_to_deboard", "driver_wants_to_onboard", "dsk_location",
        "ic_location", "is_driver_activate", "is_driver_deactivate",
        "onboarding_status", "partner_station_location", "partner_station_swap_process",
        "penalty_reason", "swap_history", "swap_price_inquiry", "swap_process",
        "user_active_plan_details", "wallet_balance", "why_choose_battery_smart"
    }
    
    # Load raw data
    print(f"Loading raw training data from: {raw_data_path}")
    raw_data = load_data(raw_data_path)
    print(f"Loaded {len(raw_data):,} samples")
    
    # Filter to only valid 19 intents
    print(f"\nFiltering to {len(VALID_INTENTS)} valid intents...")
    raw_data = [item for item in raw_data if item.get('intent') in VALID_INTENTS]
    print(f"After filtering: {len(raw_data):,} samples")
    
    # Validate data
    print("\nValidating data...")
    valid_data, issues = validate_data(raw_data)
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    print(f"Valid samples: {len(valid_data)}")
    
    # Create label mapping
    print("\nCreating label mapping...")
    label_mapping = create_label_mapping(valid_data)
    print(f"Number of labels: {len(label_mapping['label2id'])}")
    print("Labels:", list(label_mapping['label2id'].keys()))
    
    # Add labels to data
    valid_data = add_labels_to_data(valid_data, label_mapping)
    
    # Split data
    print("\nSplitting data into train/val/test...")
    train_data, val_data, test_data = stratified_split(valid_data)
    # Upsample low-frequency intents in training set only
    train_data = upsample_train_set(train_data, min_samples_per_intent=50)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Verify all intents are represented
    train_intents = set(item['intent'] for item in train_data)
    val_intents = set(item['intent'] for item in val_data)
    test_intents = set(item['intent'] for item in test_data)
    all_intents = set(label_mapping['label2id'].keys())
    
    print(f"\nIntents in train: {len(train_intents)}")
    print(f"Intents in val: {len(val_intents)}")
    print(f"Intents in test: {len(test_intents)}")
    
    missing_in_train = all_intents - train_intents
    missing_in_val = all_intents - val_intents
    missing_in_test = all_intents - test_intents
    
    if missing_in_train:
        print(f"WARNING: Intents missing from train: {missing_in_train}")
    if missing_in_val:
        print(f"WARNING: Intents missing from val: {missing_in_val}")
    if missing_in_test:
        print(f"WARNING: Intents missing from test: {missing_in_test}")
    
    # Save processed data
    print("\nSaving processed data...")
    
    # Save as JSON Lines (for Hugging Face datasets)
    save_jsonl(train_data, processed_dir / "train.json")
    save_jsonl(val_data, processed_dir / "val.json")
    save_jsonl(test_data, processed_dir / "test.json")
    
    # Save as regular JSON for reference
    save_json(valid_data, processed_dir / "valid_data.json")
    
    # Save label mapping
    with open(processed_dir / "label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Print distribution in each split
    print("\nIntent distribution in splits:")
    print("\nTrain set:")
    train_dist = Counter(item['intent'] for item in train_data)
    for intent, count in sorted(train_dist.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count}")
    
    print("\nVal set:")
    val_dist = Counter(item['intent'] for item in val_data)
    for intent, count in sorted(val_dist.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count}")
    
    print("\nTest set:")
    test_dist = Counter(item['intent'] for item in test_data)
    for intent, count in sorted(test_dist.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count}")
    
    # Save stats
    stats = {
        "total_samples": len(valid_data),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "num_intents": len(label_mapping['label2id']),
        "intents": list(label_mapping['label2id'].keys()),
        "train_distribution": dict(train_dist),
        "val_distribution": dict(val_dist),
        "test_distribution": dict(test_dist)
    }
    
    with open(processed_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*50)
    print("Data processing complete!")
    print(f"Output directory: {processed_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
