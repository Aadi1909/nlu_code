#!/usr/bin/env python3
"""
Data Validation Script for Intent Classification Training

This script validates:
1. All intents in training data match config/intents.yaml
2. All entities in training data match config/entities.yaml  
3. Data format is correct for training
4. No duplicate or empty entries
5. Label mapping consistency
"""

import json
import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple
import sys


def load_json(filepath: str) -> list:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(filepath: str) -> dict:
    """Load YAML file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_intents_from_config(config_path: str) -> Set[str]:
    """Extract all intent names from intents.yaml config."""
    config = load_yaml(config_path)
    intents = set()
    
    for intent_config in config.get("intents", []):
        intent_name = intent_config.get("intent")
        if intent_name:
            intents.add(intent_name)
    
    return intents


def get_entities_from_config(config_path: str) -> Set[str]:
    """Extract all entity names from entities.yaml config."""
    config = load_yaml(config_path)
    entities = set()
    
    for entity_config in config.get("entities", []):
        entity_name = entity_config.get("entity")
        if entity_name:
            entities.add(entity_name)
    
    return entities


def get_intents_from_data(data: list) -> Set[str]:
    """Extract all unique intents from training data."""
    return set(item['intent'] for item in data if 'intent' in item)


def get_entities_from_data(data: list) -> Set[str]:
    """Extract all unique entity types from training data."""
    entities = set()
    for item in data:
        for entity in item.get('entities', []):
            if entity.get('entity'):
                entities.add(entity['entity'])
    return entities


def validate_data_format(data: list) -> Tuple[int, List[str]]:
    """Validate each data item has required fields."""
    issues = []
    valid_count = 0
    
    for idx, item in enumerate(data):
        item_issues = []
        
        # Check required fields
        if 'text' not in item or not item['text']:
            item_issues.append("missing/empty 'text'")
        elif not isinstance(item['text'], str):
            item_issues.append("'text' is not a string")
        
        if 'intent' not in item or not item['intent']:
            item_issues.append("missing/empty 'intent'")
        elif not isinstance(item['intent'], str):
            item_issues.append("'intent' is not a string")
        
        if item_issues:
            issues.append(f"Item {idx}: {', '.join(item_issues)}")
        else:
            valid_count += 1
    
    return valid_count, issues


def check_duplicates(data: list) -> Dict:
    """Check for duplicate texts in the data."""
    text_counts = Counter(item.get('text', '') for item in data)
    duplicates = {text: count for text, count in text_counts.items() if count > 1 and text}
    return duplicates


def validate_label_mapping(data: list, label_mapping_path: str) -> List[str]:
    """Validate that label mapping covers all intents in data."""
    issues = []
    
    if not Path(label_mapping_path).exists():
        issues.append(f"Label mapping file not found: {label_mapping_path}")
        return issues
    
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    data_intents = get_intents_from_data(data)
    mapping_intents = set(label_mapping.get('label2id', {}).keys())
    
    missing_in_mapping = data_intents - mapping_intents
    extra_in_mapping = mapping_intents - data_intents
    
    if missing_in_mapping:
        issues.append(f"Intents in data but NOT in label_mapping: {missing_in_mapping}")
    
    if extra_in_mapping:
        issues.append(f"Intents in label_mapping but NOT in data: {extra_in_mapping}")
    
    return issues


def main():
    print("=" * 70)
    print("DATA VALIDATION FOR INTENT CLASSIFICATION TRAINING")
    print("=" * 70)
    
    # Paths
    base_dir = Path(__file__).parent
    config_dir = base_dir / "config"
    data_dir = base_dir / "data"
    
    intents_config_path = config_dir / "intents.yaml"
    entities_config_path = config_dir / "entities.yaml"
    
    # Find the training data file
    training_data_files = [
        data_dir / "raw" / "training_data_cleaned.json",
        data_dir / "raw" / "training_data.json",
    ]
    
    training_data_path = None
    for path in training_data_files:
        if path.exists():
            training_data_path = path
            break
    
    if not training_data_path:
        print("ERROR: No training data file found!")
        sys.exit(1)
    
    print(f"\n📂 Using training data: {training_data_path}")
    
    # Load data
    print("\n1️⃣  Loading data...")
    training_data = load_json(training_data_path)
    print(f"   Total records: {len(training_data):,}")
    
    # Load configs
    config_intents = get_intents_from_config(intents_config_path) if intents_config_path.exists() else set()
    config_entities = get_entities_from_config(entities_config_path) if entities_config_path.exists() else set()
    
    print(f"   Config intents: {len(config_intents)}")
    print(f"   Config entities: {len(config_entities)}")
    
    # Get intents and entities from data
    data_intents = get_intents_from_data(training_data)
    data_entities = get_entities_from_data(training_data)
    
    print(f"   Data intents: {len(data_intents)}")
    print(f"   Data entities: {len(data_entities)}")
    
    all_issues = []
    
    # 2. Validate data format
    print("\n2️⃣  Validating data format...")
    valid_count, format_issues = validate_data_format(training_data)
    print(f"   Valid records: {valid_count:,} / {len(training_data):,}")
    if format_issues:
        print(f"   ⚠️  Format issues found: {len(format_issues)}")
        for issue in format_issues[:5]:
            print(f"      - {issue}")
        if len(format_issues) > 5:
            print(f"      ... and {len(format_issues) - 5} more")
        all_issues.extend(format_issues)
    else:
        print("   ✅ All records have valid format")
    
    # 3. Check intent consistency
    print("\n3️⃣  Checking intent consistency...")
    if config_intents:
        missing_from_config = data_intents - config_intents
        missing_from_data = config_intents - data_intents
        
        if missing_from_config:
            print(f"   ⚠️  Intents in DATA but NOT in config/intents.yaml:")
            for intent in sorted(missing_from_config):
                count = sum(1 for item in training_data if item.get('intent') == intent)
                print(f"      - {intent} ({count:,} samples)")
            all_issues.append(f"Intents missing from config: {missing_from_config}")
        
        if missing_from_data:
            print(f"   ⚠️  Intents in CONFIG but NOT in training data:")
            for intent in sorted(missing_from_data):
                print(f"      - {intent}")
            all_issues.append(f"Intents missing from data: {missing_from_data}")
        
        if not missing_from_config and not missing_from_data:
            print("   ✅ All intents match between config and data")
    else:
        print("   ⚠️  Could not load intents.yaml config")
    
    # 4. Check entity consistency
    print("\n4️⃣  Checking entity consistency...")
    if config_entities:
        missing_from_config = data_entities - config_entities
        missing_from_data = config_entities - data_entities
        
        if missing_from_config:
            print(f"   ⚠️  Entities in DATA but NOT in config/entities.yaml:")
            for entity in sorted(missing_from_config):
                print(f"      - {entity}")
            all_issues.append(f"Entities missing from config: {missing_from_config}")
        
        if missing_from_data:
            print(f"   ℹ️  Entities in CONFIG but NOT in training data (OK if not all used):")
            for entity in sorted(missing_from_data):
                print(f"      - {entity}")
        
        if not missing_from_config:
            print("   ✅ All entities in data are defined in config")
    else:
        print("   ⚠️  Could not load entities.yaml config")
    
    # 5. Check for duplicates
    print("\n5️⃣  Checking for duplicates...")
    duplicates = check_duplicates(training_data)
    if duplicates:
        total_dups = sum(count - 1 for count in duplicates.values())
        print(f"   ⚠️  Found {len(duplicates)} texts with duplicates ({total_dups} extra copies)")
        # Show top duplicates
        top_dups = sorted(duplicates.items(), key=lambda x: -x[1])[:5]
        for text, count in top_dups:
            preview = text[:50] + "..." if len(text) > 50 else text
            print(f"      - '{preview}' appears {count} times")
    else:
        print("   ✅ No duplicate texts found")
    
    # 6. Check label mapping
    print("\n6️⃣  Checking label mapping consistency...")
    label_mapping_path = data_dir / "processed" / "label_mapping.json"
    if label_mapping_path.exists():
        mapping_issues = validate_label_mapping(training_data, label_mapping_path)
        if mapping_issues:
            for issue in mapping_issues:
                print(f"   ⚠️  {issue}")
            all_issues.extend(mapping_issues)
        else:
            print("   ✅ Label mapping is consistent with data")
    else:
        print("   ⚠️  label_mapping.json not found - run process_training_data.py first")
    
    # 7. Intent distribution
    print("\n7️⃣  Intent distribution:")
    intent_counts = Counter(item.get('intent') for item in training_data)
    total = len(training_data)
    
    # Check for imbalance
    min_count = min(intent_counts.values())
    max_count = max(intent_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"   Min samples per intent: {min_count:,}")
    print(f"   Max samples per intent: {max_count:,}")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 100:
        print("   ⚠️  Severe class imbalance detected! Consider data augmentation.")
        all_issues.append(f"Severe class imbalance: {imbalance_ratio:.1f}x")
    elif imbalance_ratio > 10:
        print("   ⚠️  Moderate class imbalance. May affect model performance.")
    
    print("\n   Top 10 intents:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:10]:
        pct = (count / total) * 100
        print(f"      {intent}: {count:,} ({pct:.1f}%)")
    
    print("\n   Bottom 5 intents (may need more data):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1])[:5]:
        pct = (count / total) * 100
        print(f"      {intent}: {count:,} ({pct:.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    if all_issues:
        print(f"⚠️  VALIDATION COMPLETED WITH {len(all_issues)} ISSUE(S)")
        print("\nCritical issues to fix before training:")
        for i, issue in enumerate(all_issues[:10], 1):
            print(f"   {i}. {issue}")
        print("\n💡 Recommendation: Fix these issues before running training")
    else:
        print("✅ VALIDATION PASSED - DATA IS READY FOR TRAINING")
    print("=" * 70)
    
    # Return intents for potential config update
    print(f"\n📋 All intents in training data ({len(data_intents)}):")
    for intent in sorted(data_intents):
        print(f"   - {intent}")
    
    return len(all_issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
