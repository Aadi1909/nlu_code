#!/usr/bin/env python3
"""
Comprehensive script to improve model accuracy:
1. Augment training data
2. Fix data inconsistencies
3. Generate more training examples from intent configs
4. Retrain with better hyperparameters
"""

import json
import yaml
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import pandas as pd

# Augmentation patterns
AUGMENTATION_PATTERNS = {
    "swap": ["swap", "battery swap", "battery change", "battery exchange", "battery badalna"],
    "station": ["station", "swap station", "battery station", "kendra", "center"],
    "battery": ["battery", "baitri", "cell", "charge"],
    "subscription": ["subscription", "plan", "package", "membership", "yojana"],
    "payment": ["payment", "pay", "paise", "paisa", "bhugtan"],
    "agent": ["agent", "representative", "customer care", "support", "helper"],
}

HINGLISH_VARIATIONS = {
    "kya": ["kya", "kya hai", "kaun sa"],
    "mere": ["mere", "mera", "meri"],
    "kahan": ["kahan", "kaha", "kidhar"],
    "batao": ["batao", "bataiye", "batana", "batayein"],
    "chahiye": ["chahiye", "chahie", "chaiye"],
}

class DataAugmenter:
    """Augment training data to improve model accuracy"""
    
    def __init__(self, config_dir: str = "../config"):
        self.config_dir = Path(config_dir)
        self.intents_config = self._load_intents_config()
        
    def _load_intents_config(self):
        """Load intents configuration"""
        with open(self.config_dir / "intents.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def generate_from_config(self) -> List[Dict]:
        """Generate training examples from intent config"""
        examples = []
        label_mapping = {}
        
        for idx, intent_data in enumerate(self.intents_config['intents']):
            intent = intent_data['intent']
            label_mapping[intent] = idx
            
            # Collect all examples from config
            for lang in ['hindi', 'english', 'hinglish']:
                if lang in intent_data.get('examples', {}):
                    for text in intent_data['examples'][lang]:
                        examples.append({
                            'text': text,
                            'intent': intent,
                            'language': lang,
                            'label': idx
                        })
        
        return examples, label_mapping
    
    def augment_text(self, text: str, language: str) -> List[str]:
        """Apply augmentation to a single text"""
        augmented = [text]
        
        # Hinglish-specific augmentations
        if language == 'hinglish':
            for word, variations in HINGLISH_VARIATIONS.items():
                if word in text.lower():
                    for var in variations:
                        new_text = re.sub(word, var, text, flags=re.IGNORECASE)
                        if new_text != text:
                            augmented.append(new_text)
        
        # Add common variations
        variations = [
            text.lower(),
            text.capitalize(),
        ]
        
        # Add please/polite forms
        polite_additions = [
            f"{text} please",
            f"please {text}",
            f"{text} karo",
            f"mujhe {text}",
        ]
        
        augmented.extend([v for v in variations + polite_additions if v not in augmented])
        
        return augmented[:5]  # Limit to 5 variations per original
    
    def augment_dataset(self, data: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """Augment entire dataset"""
        augmented_data = []
        
        for item in data:
            # Keep original
            augmented_data.append(item)
            
            # Generate variations
            variations = self.augment_text(item['text'], item.get('language', 'hinglish'))
            
            for var in variations[:augmentation_factor]:
                if var != item['text']:
                    augmented_data.append({
                        **item,
                        'text': var
                    })
        
        return augmented_data
    
    def balance_dataset(self, data: List[Dict], min_samples_per_class: int = 30) -> List[Dict]:
        """Balance dataset by oversampling minority classes"""
        # Group by intent
        intent_groups = defaultdict(list)
        for item in data:
            intent_groups[item['intent']].append(item)
        
        balanced_data = []
        
        for intent, items in intent_groups.items():
            balanced_data.extend(items)
            
            # Oversample if needed
            if len(items) < min_samples_per_class:
                additional_needed = min_samples_per_class - len(items)
                oversampled = random.choices(items, k=additional_needed)
                
                # Apply slight modifications to oversampled items
                for item in oversampled:
                    # Add variation
                    variations = self.augment_text(item['text'], item.get('language', 'hinglish'))
                    if len(variations) > 1:
                        modified_item = item.copy()
                        modified_item['text'] = random.choice(variations[1:])
                        balanced_data.append(modified_item)
                    else:
                        balanced_data.append(item)
        
        return balanced_data
    
    def add_missing_intents(self, data: List[Dict]) -> List[Dict]:
        """Add examples for intents missing from data"""
        # Common templates for missing intents
        templates = {
            "swap_history": [
                "meri swap history dikhao",
                "kitne swap kiye maine",
                "swap record chahiye",
                "previous swaps batao",
                "purane swaps ki list",
            ],
            "help_general": [
                "help chahiye",
                "mujhe help karo",
                "kuch samajh nahi aa raha",
                "guide karo",
                "assistance chahiye",
            ],
            "bye": [
                "bye",
                "goodbye",
                "alvida",
                "chalo baad me baat karte hain",
                "thanks bye",
            ],
            "greet": [
                "hello",
                "hi",
                "namaste",
                "hey",
                "good morning",
            ],
            "thank": [
                "thank you",
                "thanks",
                "dhanyavaad",
                "shukriya",
                "bahut shukriya",
            ],
        }
        
        # Get existing intents
        existing_intents = {item['intent'] for item in data}
        
        # Find label for each intent
        label_mapping = {item['intent']: item['label'] for item in data}
        
        # Add missing examples
        additional_data = []
        for intent, examples in templates.items():
            if intent in label_mapping:
                for text in examples:
                    additional_data.append({
                        'text': text,
                        'intent': intent,
                        'language': 'hinglish',
                        'label': label_mapping[intent]
                    })
        
        return data + additional_data


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSON Lines file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], filepath: Path):
    """Save to JSON Lines file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    print("="*60)
    print("Data Augmentation & Model Improvement")
    print("="*60)
    
    # Paths
    data_dir = Path("../data/processed")
    
    # Load existing data
    print("\n1. Loading existing data...")
    train_data = load_jsonl(data_dir / "train.json")
    val_data = load_jsonl(data_dir / "val.json")
    test_data = load_jsonl(data_dir / "test.json")
    
    print(f"   Original sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Initialize augmenter
    augmenter = DataAugmenter()
    
    # Generate examples from config
    print("\n2. Generating examples from intent config...")
    config_examples, label_mapping = augmenter.generate_from_config()
    print(f"   Generated {len(config_examples)} examples from config")
    
    # Merge with existing data
    all_train_data = train_data + [ex for ex in config_examples if ex not in train_data]
    
    # Add missing intents
    print("\n3. Adding examples for underrepresented intents...")
    all_train_data = augmenter.add_missing_intents(all_train_data)
    print(f"   Total examples after adding missing: {len(all_train_data)}")
    
    # Balance dataset
    print("\n4. Balancing dataset...")
    balanced_data = augmenter.balance_dataset(all_train_data, min_samples_per_class=30)
    print(f"   Balanced dataset size: {len(balanced_data)}")
    
    # Augment with variations
    print("\n5. Applying data augmentation...")
    augmented_train = augmenter.augment_dataset(balanced_data, augmentation_factor=2)
    print(f"   Augmented training size: {len(augmented_train)}")
    
    # Shuffle
    random.shuffle(augmented_train)
    
    # Split into train/val
    total_size = len(augmented_train)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    final_train = augmented_train[:train_size]
    final_val = augmented_train[train_size:train_size + val_size]
    final_test = augmented_train[train_size + val_size:]
    
    # Save augmented data
    print("\n6. Saving augmented datasets...")
    save_jsonl(final_train, data_dir / "train_augmented.json")
    save_jsonl(final_val, data_dir / "val_augmented.json")
    save_jsonl(final_test, data_dir / "test_augmented.json")
    
    print(f"   Saved:")
    print(f"   - Train: {len(final_train)} examples")
    print(f"   - Val: {len(final_val)} examples")
    print(f"   - Test: {len(final_test)} examples")
    
    # Generate statistics
    intent_counts = defaultdict(int)
    for item in final_train:
        intent_counts[item['intent']] += 1
    
    print("\n7. Dataset Statistics:")
    print(f"   Total training examples: {len(final_train)}")
    print(f"   Number of intents: {len(intent_counts)}")
    print(f"\n   Examples per intent:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {intent}: {count}")
    
    print("\n" + "="*60)
    print("✅ Data augmentation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review augmented data: data/processed/*_augmented.json")
    print("2. Retrain model with: python3 train_intent_classifier.py --use-augmented")
    print("="*60)


if __name__ == "__main__":
    main()
