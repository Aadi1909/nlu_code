

import random
import re
from typing import List, Dict
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

class DataAugmenter:
    """Augment training data to handle class imbalance"""
    
    def __init__(self):
        # Initialize augmenters
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        self.char_aug = nac.KeyboardAug()
        
    def augment_minority_classes(self, 
                                 data: List[Dict], 
                                 target_samples: int = 500,
                                 threshold: int = 300) -> List[Dict]:
        """
        Augment intents with fewer than threshold samples
        """
        # Group by intent
        intent_groups = {}
        for item in data:
            intent = item['intent']
            if intent not in intent_groups:
                intent_groups[intent] = []
            intent_groups[intent].append(item)
        
        augmented_data = data.copy()
        
        for intent, examples in intent_groups.items():
            if len(examples) < threshold:
                needed = target_samples - len(examples)
                print(f"Augmenting {intent}: {len(examples)} -> {target_samples}")
                
                augmented_examples = self._augment_examples(examples, needed)
                augmented_data.extend(augmented_examples)
        
        return augmented_data
    
    def _augment_examples(self, examples: List[Dict], num_augment: int) -> List[Dict]:
        """Generate augmented examples"""
        augmented = []
        
        for i in range(num_augment):
            # Randomly select an example to augment
            source_example = random.choice(examples)
            
            # Choose augmentation strategy
            strategy = random.choice(['synonym', 'backtranslation', 'paraphrase'])
            
            if strategy == 'synonym':
                aug_example = self._synonym_augment(source_example)
            elif strategy == 'backtranslation':
                aug_example = self._simple_paraphrase(source_example)
            else:
                aug_example = self._simple_paraphrase(source_example)
            
            augmented.append(aug_example)
        
        return augmented
    
    def _synonym_augment(self, example: Dict) -> Dict:
        """Replace words with synonyms"""
        try:
            augmented_text = self.synonym_aug.augment(example['text'])
            
            # Create new example
            new_example = example.copy()
            new_example['text'] = augmented_text
            new_example['augmented'] = True
            
            # Note: Entity positions may be invalid after augmentation
            # For production, implement proper entity alignment
            if 'entities' in new_example:
                new_example['entities'] = []  # Clear entities for now
            
            return new_example
        except:
            return example.copy()
    
    def _simple_paraphrase(self, example: Dict) -> Dict:
        """Simple paraphrasing rules"""
        text = example['text']
        
        # Simple replacements for common phrases
        replacements = {
            'where is': 'where can I find',
            'what is': "what's",
            'check': 'verify',
            'tell me': 'show me',
            'my': 'mera',
            'battery': 'battery level',
        }
        
        for old, new in replacements.items():
            if random.random() > 0.5:  # 50% chance
                text = text.replace(old, new)
        
        new_example = example.copy()
        new_example['text'] = text
        new_example['augmented'] = True
        
        return new_example


# Usage Script
if __name__ == "__main__":
    # Step 1: Validate data
    from data_validator import DataValidator
    
    validator = DataValidator()
    
    with open('data/raw/training_data.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    valid_data, invalid_data = validator.validate_dataset(raw_data)
    
    # Step 2: Split data
    loader = DataLoader('data/processed/valid_data.json')
    train_data, val_data, test_data = loader.load_and_split(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    )
    
    # Step 3: Augment training data
    augmenter = DataAugmenter()
    augmented_train = augmenter.augment_minority_classes(
        train_data,
        target_samples=500,
        threshold=300
    )
    
    # Step 4: Save processed data
    loader.save_splits(augmented_train, val_data, test_data)
    
    print("\n✅ Data preprocessing complete!")