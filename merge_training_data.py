#!/usr/bin/env python3
"""
Script to merge new training data with existing training data.
This ensures that all new intents are added to the main training dataset.
"""

import json
import os
from pathlib import Path

def load_json_file(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, filepath, backup=True):
    """Save JSON data to file with optional backup."""
    if backup and os.path.exists(filepath):
        backup_path = f"{filepath}.backup"
        print(f"Creating backup at: {backup_path}")
        with open(filepath, 'r', encoding='utf-8') as f:
            backup_data = f.read()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(backup_data)
    
    print(f"Saving data to: {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_training_data(existing_file, new_file, output_file=None):
    """
    Merge new training data with existing data.
    
    Args:
        existing_file: Path to existing training data
        new_file: Path to new training data to add
        output_file: Optional output file (defaults to existing_file)
    """
    # Load both datasets
    print(f"Loading existing data from: {existing_file}")
    existing_data = load_json_file(existing_file)
    print(f"Found {len(existing_data)} existing examples")
    
    print(f"\nLoading new data from: {new_file}")
    new_data = load_json_file(new_file)
    print(f"Found {len(new_data)} new examples")
    
    # Get unique intents from new data
    new_intents = set(item['intent'] for item in new_data)
    print(f"\nNew intents being added: {sorted(new_intents)}")
    
    # Merge data
    merged_data = existing_data + new_data
    print(f"\nTotal examples after merge: {len(merged_data)}")
    
    # Statistics
    intent_counts = {}
    for item in merged_data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nIntent distribution:")
    for intent in sorted(intent_counts.keys()):
        print(f"  {intent}: {intent_counts[intent]} examples")
    
    # Save merged data
    output_path = output_file or existing_file
    save_json_file(merged_data, output_path, backup=True)
    
    print(f"\n✓ Successfully merged training data!")
    print(f"✓ Total examples: {len(merged_data)}")
    print(f"✓ Total unique intents: {len(intent_counts)}")
    
    return merged_data

def main():
    """Main function."""
    # Define paths
    base_dir = Path(__file__).parent
    existing_file = base_dir / "data" / "raw" / "training_data.json"
    new_file = base_dir / "data" / "raw" / "new_intents_training_data.json"
    
    # Check if files exist
    if not existing_file.exists():
        print(f"ERROR: Existing training data not found at: {existing_file}")
        return
    
    if not new_file.exists():
        print(f"ERROR: New training data not found at: {new_file}")
        return
    
    # Merge data
    print("=" * 70)
    print("MERGING TRAINING DATA")
    print("=" * 70)
    merge_training_data(str(existing_file), str(new_file))
    print("=" * 70)

if __name__ == "__main__":
    main()
