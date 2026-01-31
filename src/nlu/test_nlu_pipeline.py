import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from nlu.nlu_pipeline import NLUPipeline, DialogueManager
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_turn():
    """Test single-turn predictions"""
    
    logger.info("=" * 80)
    logger.info("TEST 1: SINGLE-TURN PREDICTIONS")
    logger.info("=" * 80)
    
    # Initialize pipeline
    nlu = NLUPipeline()
    
    # Test cases
    test_cases = [
        {
            'text': 'where is my battery swap station',
            'context': {'driver_id': 'DL123456'}
        },
        {
            'text': 'check booking 12345678 for driver AB123456',
            'context': None
        },
        {
            'text': 'what is my battery health',
            'context': {'driver_id': 'DL123456', 'vehicle_id': 'DL01AA1234'}
        },
        {
            'text': 'nearest swap station to Koramangala',
            'context': None
        },
        {
            'text': 'mera battery 20% hai aur swap karna hai',
            'context': {'driver_id': 'DL123456'}
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n--- Test Case {i} ---")
        logger.info(f"Input: {test['text']}")
        logger.info(f"Context: {test['context']}")
        
        result = nlu.process(test['text'], test['context'])
        
        logger.info(f"\nResults:")
        logger.info(f"  Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        logger.info(f"  Entities: {len(result['entities'])}")
        for entity in result['entities']:
            logger.info(f"    - {entity['entity']}: {entity['value']}")
        logger.info(f"  Slots filled: {len(result['slots'])}/{len(result['required_slots'])}")
        logger.info(f"  Complete: {result['is_complete']}")
        logger.info(f"  Missing: {result['missing_slots']}")
        logger.info(f"  Action: {result['next_action']}")
        
        # Save detailed result
        with open(f'test_results/test_case_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def test_multi_turn():
    """Test multi-turn conversations"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: MULTI-TURN CONVERSATION")
    logger.info("=" * 80)
    
    # Initialize
    nlu = NLUPipeline()
    dm = DialogueManager(nlu)
    
    # Simulate conversation
    conversations = [
        {
            'session_id': 'session_001',
            'context': {'driver_id': 'DL123456'},
            'turns': [
                'I want to check my booking',
                '12345678',  # Providing booking_id
            ]
        },
        {
            'session_id': 'session_002',
            'context': None,
            'turns': [
                'nearest swap station',
                'Koramangala',  # Providing location
            ]
        }
    ]
    
    for conv_idx, conversation in enumerate(conversations, 1):
        logger.info(f"\n--- Conversation {conv_idx} ---")
        logger.info(f"Session: {conversation['session_id']}")
        logger.info(f"Context: {conversation['context']}")
        
        for turn_idx, user_input in enumerate(conversation['turns'], 1):
            logger.info(f"\n  Turn {turn_idx}:")
            logger.info(f"    User: {user_input}")
            
            response = dm.process_turn(
                text=user_input,
                session_id=conversation['session_id'],
                context=conversation['context']
            )
            
            logger.info(f"    Bot: {response['text']}")
            logger.info(f"    Action: {response['action']}")


def test_edge_cases():
    """Test edge cases and error handling"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: EDGE CASES")
    logger.info("=" * 80)
    
    nlu = NLUPipeline()
    
    edge_cases = [
        'asdfghjkl',  # Gibberish
        '12345',  # Only numbers
        'hello',  # Generic greeting (out of scope)
        'my battery is not working properly and I am facing issues',  # Complaint
        '',  # Empty string
    ]
    
    for i, text in enumerate(edge_cases, 1):
        logger.info(f"\n--- Edge Case {i} ---")
        logger.info(f"Input: '{text}'")
        
        try:
            result = nlu.process(text)
            logger.info(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
            logger.info(f"Requires agent: {result['requires_agent']}")
            logger.info(f"Action: {result['next_action']}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")


def main():
    """Run all tests"""
    
    import os
    os.makedirs('test_results', exist_ok=True)
    
    # Check if models exist
    if not Path('models/intent_classifier').exists():
        logger.error("Models not found! Please train models first:")
        logger.error("python src/training/train_all_models.py")
        return
    
    try:
        # Run tests
        test_single_turn()
        test_multi_turn()
        test_edge_cases()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS COMPLETED")
        logger.info("=" * 80)
        logger.info("\nTest results saved to test_results/")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()