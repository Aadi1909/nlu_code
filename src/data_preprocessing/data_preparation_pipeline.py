import json
import random
import os
import sys
from typing import List, Dict, Tuple
from collections import Counter
import re
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate and clean training data"""
    
    def __init__(self):
        self.valid_intents = [
            'battery_swap_status', 'battery_health', 'nearest_swap_station',
            'booking_status', 'payment_status', 'account_balance',
            'vehicle_status', 'complaint', 'out_of_scope'
        ]
        
        self.valid_entities = [
            'driver_id', 'vehicle_id', 'booking_id', 'location',
            'battery_level', 'transaction_id', 'date', 'time', 'amount'
        ]
        
        self.errors = []
        
    def validate_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Validate entire dataset"""
        valid_data = []
        invalid_data = []
        
        logger.info(f"Validating {len(data)} examples...")
        
        for idx, example in enumerate(data):
            errors = self.validate_example(example)
            
            if errors:
                self.errors.extend([f"Example {idx}: {err}" for err in errors])
                invalid_data.append({
                    'example': example,
                    'errors': errors,
                    'index': idx
                })
            else:
                valid_data.append(example)
        
        logger.info(f"Valid: {len(valid_data)}, Invalid: {len(invalid_data)}")
        
        return valid_data, invalid_data
    
    def validate_example(self, example: Dict) -> List[str]:
        """Validate single example"""
        errors = []
        
        # Check required fields
        if 'text' not in example or not example['text'].strip():
            errors.append("Missing or empty 'text' field")
        
        if 'intent' not in example:
            errors.append("Missing 'intent' field")
        elif example['intent'] not in self.valid_intents:
            errors.append(f"Invalid intent: {example['intent']}")
        
        # Check text quality
        if 'text' in example:
            text = example['text']
            
            if len(text.strip()) < 3:
                errors.append("Text too short (< 3 characters)")
            
            if text.replace(' ', '').isdigit():
                errors.append("Text contains only numbers")
        
        # Validate entities
        if 'entities' in example and example['entities']:
            for entity in example['entities']:
                if 'entity' not in entity:
                    errors.append("Entity missing 'entity' type")
                elif entity['entity'] not in self.valid_entities:
                    errors.append(f"Invalid entity type: {entity['entity']}")
                
                if 'value' not in entity or not entity['value']:
                    errors.append("Entity missing 'value'")
                
                if 'start' not in entity or 'end' not in entity:
                    errors.append("Entity missing position (start/end)")
                
                # Validate entity spans
                if all(k in entity for k in ['start', 'end', 'value']):
                    if entity['start'] >= entity['end']:
                        errors.append("Invalid entity span (start >= end)")
                    
                    if 'text' in example:
                        try:
                            extracted = example['text'][entity['start']:entity['end']]
                            if extracted != entity['value']:
                                # Allow minor differences (case, whitespace)
                                if extracted.lower().strip() != entity['value'].lower().strip():
                                    errors.append(f"Entity span mismatch: '{extracted}' != '{entity['value']}'")
                        except:
                            errors.append("Entity span out of text bounds")
        
        return errors


class DataAugmenter:
    """Augment training data"""
    
    def __init__(self):
        self.paraphrase_templates = {
            'battery_swap_status': [
                'where is {location} swap station',
                'swap station location near {location}',
                'find swap station at {location}',
            ],
            'battery_health': [
                'what is my battery level',
                'check my battery health',
                'how much battery do I have',
                'battery percentage check',
            ],
            'nearest_swap_station': [
                'nearest swap station',
                'closest battery swap location',
                'where can I swap battery nearby',
            ]
        }
    
    def augment_minority_classes(self, 
                                 data: List[Dict], 
                                 target_samples: int = 400,
                                 threshold: int = 300) -> List[Dict]:
        """Augment intents with fewer samples"""
        # Group by intent
        intent_groups = {}
        for item in data:
            intent = item['intent']
            if intent not in intent_groups:
                intent_groups[intent] = []
            intent_groups[intent].append(item)
        
        augmented_data = data.copy()
        
        for intent, examples in intent_groups.items():
            current_count = len(examples)
            if current_count < threshold:
                needed = min(target_samples - current_count, current_count * 2)
                logger.info(f"Augmenting {intent}: {current_count} -> {current_count + needed}")
                
                augmented_examples = self._augment_examples(examples, needed, intent)
                augmented_data.extend(augmented_examples)
        
        return augmented_data
    
    def _augment_examples(self, examples: List[Dict], num_augment: int, intent: str) -> List[Dict]:
        """Generate augmented examples"""
        augmented = []
        
        for i in range(num_augment):
            source_example = random.choice(examples)
            
            # Strategy: Simple word replacement and paraphrasing
            aug_example = self._simple_augment(source_example, intent)
            augmented.append(aug_example)
        
        return augmented
    
    def _simple_augment(self, example: Dict, intent: str) -> Dict:
        """Simple augmentation with word replacements"""
        text = example['text']
        
        # Common word replacements
        replacements = [
            ('where is', 'where can I find'),
            ('what is', "what's"),
            ('check', 'verify'),
            ('tell me', 'show me'),
            ('my', 'mera'),
            ('battery', 'battery level'),
            ('swap station', 'swap point'),
            ('nearest', 'closest'),
            ('booking', 'reservation'),
        ]
        
        # Apply random replacements
        for old, new in replacements:
            if old in text.lower() and random.random() > 0.5:
                text = text.replace(old, new)
        
        # Create new example
        new_example = example.copy()
        new_example['text'] = text
        new_example['augmented'] = True
        
        # Clear entities (they may be invalid after augmentation)
        if 'entities' in new_example:
            new_example['entities'] = []
        
        return new_example


class DataSplitter:
    """Split data into train/val/test"""
    
    def split_data(self, 
                   data: List[Dict],
                   train_size: float = 0.7,
                   val_size: float = 0.15,
                   test_size: float = 0.15,
                   random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data with stratification"""
        
        # Extract intents for stratification
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
        
        logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Print distribution
        self._print_distribution(train_data, "Train")
        self._print_distribution(val_data, "Validation")
        self._print_distribution(test_data, "Test")
        
        return train_data, val_data, test_data
    
    def _print_distribution(self, data: List[Dict], split_name: str):
        """Print intent distribution"""
        intents = [item['intent'] for item in data]
        intent_counts = Counter(intents)
        
        logger.info(f"\n{split_name} distribution:")
        for intent, count in sorted(intent_counts.items()):
            percentage = (count / len(data)) * 100
            logger.info(f"  {intent}: {count} ({percentage:.1f}%)")


def main():
    """Main preprocessing pipeline"""
    
    logger.info("=" * 60)
    logger.info("BATTERY SMART NLU - DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Paths
    input_path = 'data/raw/training_data.json'
    output_dir = 'data/processed'
    
    # Check if input exists
    if not os.path.exists(input_path):
        logger.error(f"❌ Input file not found: {input_path}")
        logger.error("Please place your training_data.json in data/raw/")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    logger.info("\n📂 Step 1: Loading data...")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} examples")
    
    # Step 2: Validate data
    logger.info("\n✅ Step 2: Validating data...")
    validator = DataValidator()
    valid_data, invalid_data = validator.validate_dataset(raw_data)
    
    # Save validation report
    if invalid_data:
        logger.warning(f"⚠️  Found {len(invalid_data)} invalid examples")
        with open(f'{output_dir}/invalid_data.json', 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f, ensure_ascii=False, indent=2)
        
        with open(f'{output_dir}/validation_report.txt', 'w', encoding='utf-8') as f:
            f.write("=== Data Validation Report ===\n\n")
            f.write(f"Total errors: {len(validator.errors)}\n\n")
            for error in validator.errors:
                f.write(f"- {error}\n")
        
        logger.info(f"Validation report saved to {output_dir}/validation_report.txt")
    else:
        logger.info("✅ All examples are valid!")
    
    # Save valid data
    with open(f'{output_dir}/valid_data.json', 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)
    
    # Step 3: Split data
    logger.info("\n📊 Step 3: Splitting data...")
    splitter = DataSplitter()
    train_data, val_data, test_data = splitter.split_data(valid_data)
    
    # Step 4: Augment training data
    logger.info("\n🔄 Step 4: Augmenting training data...")
    augmenter = DataAugmenter()
    augmented_train = augmenter.augment_minority_classes(
        train_data,
        target_samples=400,
        threshold=300
    )
    logger.info(f"Training data after augmentation: {len(augmented_train)}")
    
    # Step 5: Save splits
    logger.info("\n💾 Step 5: Saving processed data...")
    
    with open(f'{output_dir}/train.json', 'w', encoding='utf-8') as f:
        json.dump(augmented_train, f, ensure_ascii=False, indent=2)
    
    with open(f'{output_dir}/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(f'{output_dir}/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Data saved to {output_dir}/")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📈 PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total raw examples: {len(raw_data)}")
    logger.info(f"Valid examples: {len(valid_data)}")
    logger.info(f"Invalid examples: {len(invalid_data)}")
    logger.info(f"Training examples (augmented): {len(augmented_train)}")
    logger.info(f"Validation examples: {len(val_data)}")
    logger.info(f"Test examples: {len(test_data)}")
    logger.info("=" * 60)
    logger.info("\n✅ Preprocessing complete! Ready for training.")


if __name__ == "__main__":
    main()