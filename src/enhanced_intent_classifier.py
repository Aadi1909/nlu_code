#!/usr/bin/env python3
"""
Enhanced NLU pipeline with:
1. Rule-based fallback for specific patterns
2. Confidence threshold for agent handoff
3. Intent routing logic (Tier 1 vs Agent handoff)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import re
from pathlib import Path
from typing import Dict, Tuple

class EnhancedIntentClassifier:
    """Enhanced intent classifier with fallback rules and confidence thresholds"""
    
    # Define Tier 1 intents (can be resolved by bot)
    TIER_1_INTENTS = {
        'check_battery_status',
        'check_battery_range',
        'find_swap_station',
        'station_directions',
        'check_station_availability',
        'check_subscription',
        'swap_count',
        'swap_history',
        'check_ride_stats',
        'payment_history',
        'greet',
        'bye',
        'thank',
        'affirm',
        'deny',
    }
    
    # Intents that need agent handoff
    AGENT_HANDOFF_INTENTS = {
        'report_issue',
        'make_payment',
        'payment_inquiry',
        'subscription_renewal',
        'upgrade_subscription',
        'help_general',
        'talk_to_agent',
        'unknown',
    }
    
    # Rule-based patterns for common queries
    PATTERN_RULES = [
        # Swap station patterns
        (r'(swap\s+station|station|kendra|center)\s+(kaha|kidhar|where|location|find|search)', 'find_swap_station'),
        (r'(nearest|najdik|paas)\s+(swap|station)', 'find_swap_station'),
        (r'(station|kendra)\s+(direction|rasta|route)', 'station_directions'),
        
        # Battery patterns
        (r'(battery|baitri|charge)\s+(kitna|kya|how much|status|level)', 'check_battery_status'),
        (r'(battery|baitri)\s+(range|distance|kitne|how far)', 'check_battery_range'),
        
        # Subscription patterns
        (r'(subscription|plan|package|membership)\s+(check|status|kya|what)', 'check_subscription'),
        (r'(subscription|plan)\s+(renew|renewal|badalna)', 'subscription_renewal'),
        (r'(subscription|plan)\s+(upgrade|better|improve)', 'upgrade_subscription'),
        
        # Swap patterns
        (r'(swap|badalna)\s+(count|kitne|how many|history)', 'swap_count'),
        (r'(swap|badalna)\s+(history|pichle|previous|last)', 'swap_history'),
        
        # Payment patterns
        (r'(payment|paise|paisa|pay|bhugtan)\s+(karna|karni|make|do)', 'make_payment'),
        (r'(payment|paise)\s+(history|pichle|previous)', 'payment_history'),
        (r'(payment|paise)\s+(pending|due|baaki|problem)', 'payment_inquiry'),
        
        # Agent patterns
        (r'(agent|representative|customer\s+care|support|person|insaan)\s+(se|talk|baat)', 'talk_to_agent'),
        (r'(problem|issue|complaint|dikkat|samasya|help)', 'report_issue'),
        
        # Greetings
        (r'^(hi|hello|hey|namaste|namaskar|hii|helo)', 'greet'),
        (r'(bye|goodbye|tata|alvida|good\s+bye)', 'bye'),
        (r'(thank|thanks|dhanyavaad|shukriya)', 'thank'),
    ]
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.4):
        """
        Initialize the enhanced classifier
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for model predictions (default 0.4)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Load label mapping
        with open(self.model_path / "label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
            self.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
            self.label2id = label_mapping["label2id"]
    
    def apply_pattern_rules(self, text: str) -> Tuple[str, float]:
        """
        Apply rule-based pattern matching
        
        Returns:
            (intent, confidence) or (None, 0.0) if no match
        """
        text_lower = text.lower().strip()
        
        for pattern, intent in self.PATTERN_RULES:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return intent, 1.0  # High confidence for rule-based matches
        
        return None, 0.0
    
    def predict_with_model(self, text: str) -> Tuple[str, float, list]:
        """
        Use ML model for prediction
        
        Returns:
            (intent, confidence, top_3_predictions)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        
        pred_id = torch.argmax(probs).item()
        confidence = probs[0][pred_id].item()
        intent = self.id2label[pred_id]
        
        # Get top 3
        top_probs, top_indices = torch.topk(probs[0], 3)
        top_3 = [
            {
                "intent": self.id2label[idx.item()],
                "confidence": prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return intent, confidence, top_3
    
    def predict(self, text: str) -> Dict:
        """
        Main prediction function with fallback logic
        
        Returns:
            {
                'intent': str,
                'confidence': float,
                'method': 'rule' or 'model',
                'should_handoff': bool,
                'reason': str,
                'top_predictions': list
            }
        """
        # Step 1: Try rule-based patterns first
        rule_intent, rule_confidence = self.apply_pattern_rules(text)
        
        if rule_intent:
            # Rule matched
            should_handoff = rule_intent in self.AGENT_HANDOFF_INTENTS
            return {
                'intent': rule_intent,
                'confidence': rule_confidence,
                'method': 'rule',
                'should_handoff': should_handoff,
                'reason': 'Tier 1 - Bot can handle' if not should_handoff else 'Needs agent assistance',
                'top_predictions': [{'intent': rule_intent, 'confidence': rule_confidence}]
            }
        
        # Step 2: Use ML model
        intent, confidence, top_3 = self.predict_with_model(text)
        
        # Step 3: Apply confidence threshold
        if confidence < self.confidence_threshold:
            # Low confidence -> handoff to agent
            return {
                'intent': 'talk_to_agent',
                'confidence': confidence,
                'method': 'fallback',
                'should_handoff': True,
                'reason': f'Low confidence ({confidence:.1%}) - Agent handoff',
                'top_predictions': top_3
            }
        
        # Step 4: Check if intent requires agent handoff
        should_handoff = intent in self.AGENT_HANDOFF_INTENTS
        
        return {
            'intent': intent,
            'confidence': confidence,
            'method': 'model',
            'should_handoff': should_handoff,
            'reason': 'Tier 1 - Bot can handle' if not should_handoff else 'Needs agent assistance',
            'top_predictions': top_3
        }


def test_enhanced_classifier():
    """Test the enhanced classifier"""
    
    # Initialize
    model_path = "/mnt/sagemaker-nvme/intent_models"
    classifier = EnhancedIntentClassifier(model_path, confidence_threshold=0.4)
    
    # Test queries
    test_queries = [
        "मुझे अपना बैलेंस चेक करना है",  # Should be payment_inquiry -> agent handoff
        "I want to check my subscription",  # Tier 1
        "battery kitna hai?",  # Tier 1
        "swap station kaha hai?",  # Tier 1 - should find_swap_station
        "मुझे payment करनी है",  # Agent handoff
        "hi, I need help",  # Greet (Tier 1)
        "I have a problem with my bike",  # Agent handoff
        "thank you",  # Tier 1
        "nearest swap station",  # Tier 1 - find_swap_station
        "battery range kitna hai",  # Tier 1
    ]
    
    print("\n" + "=" * 80)
    print("ENHANCED INTENT CLASSIFIER - TEST RESULTS")
    print("=" * 80)
    
    for query in test_queries:
        result = classifier.predict(query)
        
        print(f"\n{'='*80}")
        print(f"📝 Query: '{query}'")
        print(f"   ✅ Intent: {result['intent']}")
        print(f"   🎲 Confidence: {result['confidence']:.1%} ({result['method']})")
        print(f"   {'🤖' if not result['should_handoff'] else '👤'} {result['reason']}")
        
        if result['should_handoff']:
            print(f"   ➡️  ACTION: Transfer to agent")
        else:
            print(f"   ➡️  ACTION: Bot handles")
        
        print(f"\n   Top 3 predictions:")
        for i, pred in enumerate(result['top_predictions'][:3], 1):
            print(f"      {i}. {pred['intent']:30s} ({pred['confidence']:.1%})")
    
    print("\n" + "=" * 80)
    print("✅ Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_enhanced_classifier()
