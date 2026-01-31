import json
import random
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

class DataLoader:
    """Load and split data for training"""
    
    def __init__(self, data_path: str, stratify_by_intent: bool = True):
        self.data_path = data_path
        self.stratify = stratify_by_intent
        
    def load_and_split(self, 
                       train_size: float = 0.7,
                       val_size: float = 0.15,
                       test_size: float = 0.15,
                       random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load data and split into train/val/test
        """
        # Load data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} examples")
        
        # Shuffle data
        random.seed(random_state)
        random.shuffle(data)
        
        if self.stratify:
            # Stratified split by intent
            intents = [item['intent'] for item in data]
            
            # First split: train + (val + test)
            train_data, temp_data = train_test_split(
                data,
                train_size=train_size,
                stratify=intents,
                random_state=random_state
            )
            
            # Second split: val + test
            temp_intents = [item['intent'] for item in temp_data]
            val_ratio = val_size / (val_size + test_size)
            
            val_data, test_data = train_test_split(
                temp_data,
                train_size=val_ratio,
                stratify=temp_intents,
                random_state=random_state
            )
        else:
            # Simple random split
            n = len(data)
            train_end = int(n * train_size)
            val_end = int(n * (train_size + val_size))
            
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Print class distribution
        self._print_distribution(train_data, "Train")
        self._print_distribution(val_data, "Validation")
        self._print_distribution(test_data, "Test")
        
        return train_data, val_data, test_data
    
    def _print_distribution(self, data: List[Dict], split_name: str):
        """Print intent distribution"""
        intents = [item['intent'] for item in data]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print(f"\n{split_name} distribution:")
        for intent, count in sorted(intent_counts.items()):
            percentage = (count / len(data)) * 100
            print(f"  {intent}: {count} ({percentage:.1f}%)")
    
    def save_splits(self, train_data, val_data, test_data, output_dir: str = 'data/processed'):
        """Save splits to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/train.json', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(f'{output_dir}/val.json', 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(f'{output_dir}/test.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSplits saved to {output_dir}/")


