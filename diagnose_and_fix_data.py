#!/usr/bin/env python3
"""
Diagnose and fix training data validation issues
"""
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_training_data(file_path="data/raw/training_data.json"):
    """Analyze what's causing validation failures"""
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Total examples: {len(data)}")
    
    # Analysis counters
    total = len(data)
    missing_text = 0
    missing_intent = 0
    missing_entities = 0
    null_entities = 0
    empty_text = 0
    entity_position_issues = 0
    intent_distribution = Counter()
    valid_examples = []
    invalid_examples = []
    
    for idx, example in enumerate(data):
        issues = []
        
        # Check required fields
        if 'text' not in example:
            missing_text += 1
            issues.append("missing_text")
        elif not example['text'] or example['text'].strip() == "":
            empty_text += 1
            issues.append("empty_text")
        
        if 'intent' not in example:
            missing_intent += 1
            issues.append("missing_intent")
        else:
            intent_distribution[example['intent']] += 1
        
        if 'entities' not in example:
            missing_entities += 1
            issues.append("missing_entities")
        elif example['entities'] is None:
            null_entities += 1
            issues.append("null_entities")
        else:
            # Check entity positions
            for entity in example['entities']:
                # Allow None for context-filled entities
                if entity.get('start') is None or entity.get('end') is None:
                    continue  # This is valid for context-filled entities
                
                # Check if positions are valid
                if entity['start'] < 0 or entity['end'] < 0:
                    entity_position_issues += 1
                    issues.append(f"negative_position")
                    break
                
                if entity['start'] > entity['end']:
                    entity_position_issues += 1
                    issues.append(f"start>end")
                    break
                
                # Check if entity text matches
                if 'text' in example and example['text']:
                    text_len = len(example['text'])
                    if entity['end'] > text_len:
                        entity_position_issues += 1
                        issues.append(f"position_out_of_bounds")
                        break
        
        if issues:
            invalid_examples.append({
                'index': idx,
                'text': example.get('text', '')[:100],
                'intent': example.get('intent', 'N/A'),
                'issues': issues
            })
        else:
            valid_examples.append(example)
    
    # Print analysis
    logger.info("\n" + "="*80)
    logger.info("VALIDATION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Total examples: {total}")
    logger.info(f"Valid examples: {len(valid_examples)} ({len(valid_examples)/total*100:.1f}%)")
    logger.info(f"Invalid examples: {len(invalid_examples)} ({len(invalid_examples)/total*100:.1f}%)")
    logger.info("")
    logger.info("ISSUES FOUND:")
    logger.info(f"  - Missing 'text' field: {missing_text}")
    logger.info(f"  - Empty text: {empty_text}")
    logger.info(f"  - Missing 'intent' field: {missing_intent}")
    logger.info(f"  - Missing 'entities' field: {missing_entities}")
    logger.info(f"  - Null entities: {null_entities}")
    logger.info(f"  - Entity position issues: {entity_position_issues}")
    logger.info("")
    logger.info(f"INTENT DISTRIBUTION (Top 10):")
    for intent, count in intent_distribution.most_common(10):
        logger.info(f"  - {intent}: {count}")
    logger.info("")
    
    # Show sample invalid examples
    if invalid_examples:
        logger.info("SAMPLE INVALID EXAMPLES (first 10):")
        for example in invalid_examples[:10]:
            logger.info(f"\nIndex {example['index']}:")
            logger.info(f"  Text: {example['text']}")
            logger.info(f"  Intent: {example['intent']}")
            logger.info(f"  Issues: {', '.join(example['issues'])}")
    
    logger.info("\n" + "="*80)
    
    # Check if we need to create a cleaned version
    if len(valid_examples) > 0:
        logger.info(f"\n✅ Found {len(valid_examples)} valid examples")
        logger.info(f"Creating cleaned training data file...")
        
        # Save cleaned data
        cleaned_file = "data/raw/training_data_cleaned.json"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            json.dump(valid_examples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Cleaned data saved to: {cleaned_file}")
        logger.info(f"\nTo use cleaned data, update src/data_preprocessing/data_preparation_pipeline.py:")
        logger.info(f"  Change: file_path='data/raw/training_data.json'")
        logger.info(f"  To:     file_path='data/raw/training_data_cleaned.json'")
    else:
        logger.error("❌ No valid examples found! Training data has serious issues.")
    
    return {
        'total': total,
        'valid': len(valid_examples),
        'invalid': len(invalid_examples),
        'intent_distribution': intent_distribution
    }

if __name__ == "__main__":
    results = analyze_training_data()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Valid: {results['valid']}/{results['total']} ({results['valid']/results['total']*100:.1f}%)")
    print(f"Invalid: {results['invalid']}/{results['total']} ({results['invalid']/results['total']*100:.1f}%)")
    print("="*80)
