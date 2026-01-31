import torch
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """Intent classification component"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        logger.info(f"Loading intent classifier from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load intent mappings
        with open(f'{model_path}/intent_mapping.json', 'r') as f:
            mappings = json.load(f)
            self.intent_to_id = mappings['intent_to_id']
            self.id_to_intent = {int(k): v for k, v in mappings['id_to_intent'].items()}
        
        logger.info(f"✅ Intent classifier loaded with {len(self.id_to_intent)} intents")
    
    def predict(self, text: str) -> Dict:
        """Predict intent and confidence"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=3)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'intent': self.id_to_intent[idx.item()],
                'confidence': prob.item()
            })
        
        return {
            'intent': predictions[0]['intent'],
            'confidence': predictions[0]['confidence'],
            'all_predictions': predictions
        }


class EntityExtractor:
    """Entity extraction component"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        logger.info(f"Loading entity extractor from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load entity mappings
        with open(f'{model_path}/entity_mapping.json', 'r') as f:
            mappings = json.load(f)
            self.entity_to_id = mappings['entity_to_id']
            self.id_to_entity = {int(k): v for k, v in mappings['id_to_entity'].items()}
        
        logger.info(f"✅ Entity extractor loaded with {len(self.entity_to_id)} tags")
    
    def predict(self, text: str) -> List[Dict]:
        """Extract entities from text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
        
        # Decode entities
        entities = self._decode_entities(text, predictions, offset_mapping)
        
        # Add regex-based entities
        entities.extend(self._extract_regex_entities(text))
        
        return entities
    
    def _decode_entities(self, text: str, predictions: torch.Tensor, offset_mapping) -> List[Dict]:
        """Decode BIO tags into entities"""
        entities = []
        current_entity = None
        
        for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Skip special tokens
                continue
            
            tag = self.id_to_entity[pred.item()]
            
            if tag.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = tag[2:]
                current_entity = {
                    'entity': entity_type,
                    'start': start.item(),
                    'end': end.item(),
                    'value': text[start:end],
                    'confidence': 0.9,
                    'source': 'model'
                }
            
            elif tag.startswith('I-') and current_entity:
                # Continue current entity
                current_entity['end'] = end.item()
                current_entity['value'] = text[current_entity['start']:end]
            
            elif tag == 'O' and current_entity:
                # End current entity
                entities.append(current_entity)
                current_entity = None
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _extract_regex_entities(self, text: str) -> List[Dict]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Define patterns
        patterns = {
            'driver_id': r'\b[A-Z]{2}\d{6}\b',
            'vehicle_id': r'\b[A-Z]{2}\d{2}[A-Z]{2}\d{4}\b',
            'booking_id': r'\b\d{8,12}\b',
            'transaction_id': r'\bTXN\d{10}\b',
            'battery_level': r'\b\d{1,3}%?\b',
            'amount': r'\b₹?\s?\d+\.?\d*\b'
        }
        
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    'entity': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'value': match.group(),
                    'confidence': 0.95,
                    'source': 'regex'
                })
        
        return entities


class SlotFiller:
    """Slot filling and validation component"""
    
    def __init__(self, config_path: str = 'config/intents.yaml'):
        # Load slot requirements
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.intent_config = config.get('tier_1_intents', {})
    
    def fill_slots(self, 
                   intent: str, 
                   entities: List[Dict],
                   context: Optional[Dict] = None) -> Dict:
        """Fill slots from entities and context"""
        
        slots = {}
        
        # Get slot requirements for this intent
        intent_info = self.intent_config.get(intent, {})
        required_slots = intent_info.get('required_slots', [])
        optional_slots = intent_info.get('optional_slots', [])
        
        # Fill from extracted entities
        for entity in entities:
            entity_type = entity['entity']
            if entity_type in required_slots or entity_type in optional_slots:
                # Take highest confidence entity of this type
                if entity_type not in slots or entity['confidence'] > slots[entity_type]['confidence']:
                    slots[entity_type] = {
                        'value': entity['value'],
                        'confidence': entity['confidence'],
                        'source': entity.get('source', 'unknown')
                    }
        
        # Fill from context (e.g., authenticated user info)
        if context:
            for slot_name in required_slots:
                if slot_name not in slots and slot_name in context:
                    slots[slot_name] = {
                        'value': context[slot_name],
                        'confidence': 1.0,
                        'source': 'context'
                    }
        
        # Validate slots
        validation = self._validate_slots(intent, slots, required_slots, optional_slots)
        
        return {
            'slots': slots,
            'required_slots': required_slots,
            'optional_slots': optional_slots,
            'is_complete': validation['is_complete'],
            'missing_slots': validation['missing_slots'],
            'invalid_slots': validation['invalid_slots']
        }
    
    def _validate_slots(self, intent: str, slots: Dict, required_slots: List, optional_slots: List) -> Dict:
        """Validate if all required slots are filled and valid"""
        
        missing_slots = []
        invalid_slots = []
        
        # Check required slots
        for slot_name in required_slots:
            if slot_name not in slots:
                missing_slots.append(slot_name)
            else:
                # Validate slot value
                if not self._validate_slot_value(slot_name, slots[slot_name]['value']):
                    invalid_slots.append(slot_name)
        
        return {
            'is_complete': len(missing_slots) == 0 and len(invalid_slots) == 0,
            'missing_slots': missing_slots,
            'invalid_slots': invalid_slots
        }
    
    def _validate_slot_value(self, slot_name: str, value: str) -> bool:
        """Validate individual slot value"""
        
        validators = {
            'driver_id': lambda v: bool(re.match(r'^[A-Z]{2}\d{6}$', v)),
            'vehicle_id': lambda v: bool(re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', v)),
            'booking_id': lambda v: bool(re.match(r'^\d{8,12}$', v)),
            'transaction_id': lambda v: bool(re.match(r'^TXN\d{10}$', v)),
            'battery_level': lambda v: 0 <= int(re.sub(r'%', '', v)) <= 100,
        }
        
        if slot_name in validators:
            try:
                return validators[slot_name](value)
            except:
                return False
        
        return True  # No specific validation


class NLUPipeline:
    """Complete NLU Pipeline"""
    
    def __init__(self, 
                 intent_model_path: str = 'models/intent_classifier',
                 entity_model_path: str = 'models/entity_extractor',
                 config_path: str = 'config/intents.yaml'):
        
        logger.info("Initializing NLU Pipeline...")
        
        # Initialize components
        self.intent_classifier = IntentClassifier(intent_model_path)
        self.entity_extractor = EntityExtractor(entity_model_path)
        self.slot_filler = SlotFiller(config_path)
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("✅ NLU Pipeline initialized successfully")
    
    def process(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        Process text through complete NLU pipeline
        
        Args:
            text: Input text from user
            context: Optional context (e.g., driver_id from session)
        
        Returns:
            Complete NLU result with intent, entities, slots, and actions
        """
        
        logger.info(f"Processing: '{text}'")
        
        # Step 1: Intent Classification
        intent_result = self.intent_classifier.predict(text)
        logger.info(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.3f})")
        
        # Step 2: Entity Extraction
        entities = self.entity_extractor.predict(text)
        logger.info(f"Entities: {len(entities)} found")
        
        # Step 3: Slot Filling
        slot_result = self.slot_filler.fill_slots(
            intent=intent_result['intent'],
            entities=entities,
            context=context
        )
        logger.info(f"Slots: {len(slot_result['slots'])} filled, Complete: {slot_result['is_complete']}")
        
        # Step 4: Determine next action
        action = self._determine_action(intent_result, slot_result)
        logger.info(f"Action: {action}")
        
        # Compile results
        result = {
            'text': text,
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'all_intents': intent_result['all_predictions'],
            'entities': entities,
            'slots': slot_result['slots'],
            'required_slots': slot_result['required_slots'],
            'missing_slots': slot_result['missing_slots'],
            'is_complete': slot_result['is_complete'],
            'next_action': action,
            'backend_service': self._get_backend_service(intent_result['intent']),
            'requires_agent': self._requires_agent(intent_result['intent'], intent_result['confidence'])
        }
        
        return result
    
    def _determine_action(self, intent_result: Dict, slot_result: Dict) -> str:
        """Determine what action to take next"""
        
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        
        # Check if tier-1 intent
        tier_1_intents = self.config.get('tier_1_intents', {})
        
        if intent not in tier_1_intents:
            return 'transfer_to_agent'
        
        # Check confidence threshold
        threshold = tier_1_intents[intent].get('confidence_threshold', 0.75)
        if confidence < threshold:
            return 'clarify_intent'
        
        # Check slot completion
        if not slot_result['is_complete']:
            return 'collect_missing_slots'
        
        # All good, proceed to backend
        return 'call_backend_api'
    
    def _get_backend_service(self, intent: str) -> Optional[str]:
        """Get backend service name for intent"""
        
        service_mapping = {
            'battery_swap_status': 'swap_status_api',
            'battery_health': 'battery_health_api',
            'nearest_swap_station': 'swap_station_locator_api',
            'booking_status': 'booking_api',
            'payment_status': 'payment_api',
            'account_balance': 'wallet_api'
        }
        
        return service_mapping.get(intent)
    
    def _requires_agent(self, intent: str, confidence: float) -> bool:
        """Check if conversation should be transferred to agent"""
        
        tier_2_intents = ['complaint', 'out_of_scope', 'technical_issue']
        
        if intent in tier_2_intents:
            return True
        
        if confidence < 0.5:
            return True
        
        return False


# src/nlu/dialogue_manager.py

"""
Dialogue Manager for multi-turn conversations
"""

class DialogueManager:
    """Manage multi-turn conversations with slot filling"""
    
    def __init__(self, nlu_pipeline: NLUPipeline):
        self.nlu = nlu_pipeline
        self.conversations = {}  # session_id -> state
    
    def process_turn(self, 
                     text: str, 
                     session_id: str,
                     context: Optional[Dict] = None) -> Dict:
        """
        Process a conversation turn
        
        Args:
            text: User input
            session_id: Unique session identifier
            context: Session context (driver info, etc.)
        
        Returns:
            Response with text, action, and updated state
        """
        
        # Get or create conversation state
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'intent': None,
                'slots': {},
                'turn_count': 0,
                'history': []
            }
        
        state = self.conversations[session_id]
        state['turn_count'] += 1
        state['history'].append({'user': text})
        
        # Process with NLU
        nlu_result = self.nlu.process(text, context)
        
        # Update state
        if not state['intent']:
            state['intent'] = nlu_result['intent']
        
        # Merge slots
        for slot_name, slot_data in nlu_result['slots'].items():
            state['slots'][slot_name] = slot_data
        
        # Generate response
        response = self._generate_response(nlu_result, state)
        state['history'].append({'system': response['text']})
        
        # Clean up completed conversations
        if response['action'] in ['call_backend_api', 'transfer_to_agent']:
            if state['turn_count'] > 10:  # Keep for debugging
                del self.conversations[session_id]
        
        return response
    
    def _generate_response(self, nlu_result: Dict, state: Dict) -> Dict:
        """Generate appropriate response"""
        
        action = nlu_result['next_action']
        
        if action == 'transfer_to_agent':
            return {
                'text': "I'm having trouble understanding. Let me connect you to an agent who can help.",
                'text_hindi': "मुझे समझने में कठिनाई हो रही है। मैं आपको एक एजेंट से जोड़ता हूं जो मदद कर सकता है।",
                'action': 'transfer_to_agent',
                'transfer': True
            }
        
        elif action == 'clarify_intent':
            top_intents = nlu_result['all_intents'][:2]
            return {
                'text': f"Did you mean: {top_intents[0]['intent']} or {top_intents[1]['intent']}?",
                'text_hindi': f"क्या आप पूछना चाहते हैं: {top_intents[0]['intent']} या {top_intents[1]['intent']}?",
                'action': 'clarify'
            }
        
        elif action == 'collect_missing_slots':
            missing_slot = nlu_result['missing_slots'][0]
            prompt = self._get_slot_prompt(missing_slot)
            
            return {
                'text': prompt['en'],
                'text_hindi': prompt['hi'],
                'action': 'collect_slot',
                'missing_slot': missing_slot
            }
        
        elif action == 'call_backend_api':
            return {
                'text': "Please wait, I'm fetching your information...",
                'text_hindi': "एक क्षण रुकें, मैं आपकी जानकारी प्राप्त कर रहा हूं...",
                'action': 'backend_call',
                'backend_service': nlu_result['backend_service'],
                'params': {k: v['value'] for k, v in state['slots'].items()}
            }
        
        return {
            'text': "I didn't understand that. Can you please rephrase?",
            'text_hindi': "मैं समझ नहीं पाया। क्या आप दोबारा कह सकते हैं?",
            'action': 'fallback'
        }
    
    def _get_slot_prompt(self, slot_name: str) -> Dict:
        """Get prompt for collecting slot"""
        
        prompts = {
            'driver_id': {
                'en': "Please provide your driver ID.",
                'hi': "कृपया अपना ड्राइवर आईडी बताएं।"
            },
            'vehicle_id': {
                'en': "Please provide your vehicle number.",
                'hi': "कृपया अपना वाहन नंबर बताएं।"
            },
            'booking_id': {
                'en': "Please provide your booking ID.",
                'hi': "कृपया अपना बुकिंग आईडी बताएं।"
            },
            'location': {
                'en': "Where are you located? Please provide your location.",
                'hi': "आप कहां हैं? कृपया अपना स्थान बताएं।"
            },
            'transaction_id': {
                'en': "Please provide your transaction ID.",
                'hi': "कृपया अपना लेन-देन आईडी बताएं।"
            }
        }
        
        return prompts.get(slot_name, {
            'en': f"Please provide {slot_name}.",
            'hi': f"कृपया {slot_name} बताएं।"
        })