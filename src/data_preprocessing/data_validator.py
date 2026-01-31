
import json
import re
from typing import List, Dict, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and clean training data"""
    
    def __init__(self, config_path: str = 'config/intents.yaml'):
        self.valid_intents = self._load_valid_intents(config_path)
        self.valid_entities = self._load_valid_entities(config_path)
        self.errors = []
        
    def _load_valid_intents(self, config_path: str) -> List[str]:
        """Load valid intent names from config"""
        # Define your valid intents
        return [
            'battery_swap_status',
            'battery_health',
            'nearest_swap_station',
            'booking_status',
            'payment_status',
            'account_balance',
            'vehicle_status',
            'complaint',
            'out_of_scope'
        ]
    
    def _load_valid_entities(self, config_path: str) -> List[str]:
        """Load valid entity types"""
        return [
            'driver_id',
            'vehicle_id',
            'booking_id',
            'location',
            'battery_level',
            'transaction_id',
            'date',
            'time',
            'amount'
        ]
    
    def validate_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate entire dataset
        Returns: (valid_data, invalid_data)
        """
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
        
        logger.info(f"Valid examples: {len(valid_data)}")
        logger.info(f"Invalid examples: {len(invalid_data)}")
        
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
            
            # Check minimum length
            if len(text.strip()) < 3:
                errors.append("Text too short (< 3 characters)")
            
            # Check for weird characters
            if re.search(r'[^\w\s\.,!?।-]', text, re.UNICODE):
                # Allow common punctuation
                pass
            
            # Check if text is all numbers
            if text.replace(' ', '').isdigit():
                errors.append("Text contains only numbers")
        
        # Validate entities
        if 'entities' in example and example['entities']:
            for entity in example['entities']:
                # Check required entity fields
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
                        extracted = example['text'][entity['start']:entity['end']]
                        if extracted != entity['value']:
                            errors.append(f"Entity span mismatch: '{extracted}' != '{entity['value']}'")
        
        return errors
    
    def save_validation_report(self, output_path: str):
        """Save validation errors to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== Data Validation Report ===\n\n")
            f.write(f"Total errors found: {len(self.errors)}\n\n")
            for error in self.errors:
                f.write(f"- {error}\n")
        
        logger.info(f"Validation report saved to {output_path}")


# Usage
if __name__ == "__main__":
    # Load data
    with open('data/raw/training_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate
    validator = DataValidator()
    valid_data, invalid_data = validator.validate_dataset(data)
    
    # Save results
    with open('data/processed/valid_data.json', 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)
    
    with open('data/processed/invalid_data.json', 'w', encoding='utf-8') as f:
        json.dump(invalid_data, f, ensure_ascii=False, indent=2)
    
    validator.save_validation_report('data/processed/validation_report.txt')