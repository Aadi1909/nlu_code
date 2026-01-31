#!/usr/bin/env python3
"""
Battery Smart Tier 1 Intent Classifier

Handles these intents from training_data.json (10,000 examples):
- battery_swap_status: Swap station info/address (TIER 1)
- booking_cancel: Cancel a booking (AGENT)
- booking_status: Check booking status (TIER 1)
- driver_deboarding: Close driver account (AGENT)
- driver_onboarding: New driver registration (AGENT)
- driver_onboarding_status: Check registration status (TIER 1)
- penalty_reason: Explain fines/penalties (TIER 1)
- swap_history: View past swaps (TIER 1)
- swap_process: How to swap battery (TIER 1)
- wallet_balance: Check wallet/balance (TIER 1)

TIER 1 (7 intents): Bot resolves directly
AGENT (3 intents): Handoff to human agent
"""

import re
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

# Optional: ML model imports (only loaded if model exists)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class IntentCategory(Enum):
    TIER_1 = "tier1"
    AGENT = "agent"


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: str
    confidence: float
    category: IntentCategory
    method: str  # 'rule', 'model', 'fallback'
    action: str  # What to do next
    entities: Dict = None
    top_predictions: List = None
    
    def to_dict(self) -> Dict:
        return {
            'intent': self.intent,
            'confidence': self.confidence,
            'category': self.category.value,
            'method': self.method,
            'action': self.action,
            'entities': self.entities or {},
            'top_predictions': self.top_predictions or [],
            'should_handoff': self.category == IntentCategory.AGENT
        }


class Tier1Classifier:
    """
    Efficient Tier 1 intent classifier with:
    1. Rule-based patterns for instant matching (100% confidence)
    2. ML model for complex queries (optional)
    3. Clear Tier 1 vs Agent handoff routing
    """
    
    # ==================== INTENT DEFINITIONS ====================
    
    INTENT_CONFIG = {
        # TIER 1 - Bot can handle
        'battery_swap_status': {
            'category': IntentCategory.TIER_1,
            'description': 'Swap station address/location/info',
            'action': 'Call /api/swap-stations endpoint',
            'sample': 'swap station kaha hai'
        },
        'booking_status': {
            'category': IntentCategory.TIER_1,
            'description': 'Check booking status/confirmation',
            'action': 'Call /api/booking/status endpoint',
            'sample': 'meri booking ka status'
        },
        'driver_onboarding_status': {
            'category': IntentCategory.TIER_1,
            'description': 'Check registration approval status',
            'action': 'Call /api/driver/status endpoint',
            'sample': 'mera registration status'
        },
        'penalty_reason': {
            'category': IntentCategory.TIER_1,
            'description': 'Explain fines/penalties',
            'action': 'Call /api/penalty/details endpoint',
            'sample': 'fine kyu laga'
        },
        'swap_history': {
            'category': IntentCategory.TIER_1,
            'description': 'View past battery swaps',
            'action': 'Call /api/swap/history endpoint',
            'sample': 'swap history dikhao'
        },
        'swap_process': {
            'category': IntentCategory.TIER_1,
            'description': 'How to swap battery (instructions)',
            'action': 'Return swap process guide',
            'sample': 'battery kaise swap kare'
        },
        'wallet_balance': {
            'category': IntentCategory.TIER_1,
            'description': 'Check wallet balance',
            'action': 'Call /api/wallet/balance endpoint',
            'sample': 'wallet balance check karo'
        },
        
        # AGENT HANDOFF - Need human agent
        'booking_cancel': {
            'category': IntentCategory.AGENT,
            'description': 'Cancel a booking',
            'action': 'Transfer to agent for verification',
            'sample': 'booking cancel kardo'
        },
        'driver_deboarding': {
            'category': IntentCategory.AGENT,
            'description': 'Close/deactivate driver account',
            'action': 'Transfer to agent (sensitive operation)',
            'sample': 'account close karna hai'
        },
        'driver_onboarding': {
            'category': IntentCategory.AGENT,
            'description': 'New driver registration',
            'action': 'Transfer to agent for registration',
            'sample': 'driver kaise bane'
        },
    }
    
    # ==================== RULE-BASED PATTERNS ====================
    # High-precision patterns for instant matching (ordered by specificity)
    
    PATTERN_RULES = [
        # ===== TIER 1 PATTERNS =====
        
        # Wallet/Balance
        (r'wallet\s*(status|balance|kitna|check|dekho|dikhao)', 'wallet_balance'),
        (r'(balance|paisa|paise|amount)\s*(check|kitna|kya|batao|dikhao)', 'wallet_balance'),
        (r'mere?\s*wallet\s*(mein|me|ka|ki)', 'wallet_balance'),
        (r'wallet\s*(for|ka|ki)\s*driver', 'wallet_balance'),
        (r'kitna\s*(paisa|balance|amount)', 'wallet_balance'),
        
        # Swap Station/Address
        (r'swap\s*station\s*(address|kaha|where|location|nearest|info|details)', 'battery_swap_status'),
        (r'(nearest|najdiki|pass|paas)\s*(swap|station|hub)', 'battery_swap_status'),
        (r'(station|kendra|hub)\s*(kaha|kidhar|where|address|location)', 'battery_swap_status'),
        (r'(swap|station|hub)\s*ka?\s*(address|location)', 'battery_swap_status'),
        (r'(battery\s*)?(swap|exchange)\s*(kaha|where|location)', 'battery_swap_status'),
        
        # Swap History
        (r'swap\s*(history|logs?|record|list)', 'swap_history'),
        (r'(pichle|previous|past|purane)\s*swap', 'swap_history'),
        (r'swap\s*(kitne|count|how many|total)', 'swap_history'),
        (r'(mera|mere|my)\s*swap\s*(history|record)', 'swap_history'),
        (r'swap\s*logs?\s*(dikhao|batao|show)', 'swap_history'),
        
        # Swap Process/Guide
        (r'(swap|battery)\s*(kaise|how\s*to|process|guide|tarika|procedure)', 'swap_process'),
        (r'battery\s*(badalna|change|exchange)\s*kaise', 'swap_process'),
        (r'(swap|exchange)\s*kaise\s*(kare|karu|hota)', 'swap_process'),
        (r'battery\s*swap\s*guide', 'swap_process'),
        
        # Booking Status
        (r'booking\s*(status|confirm|check|hua|hui)', 'booking_status'),
        (r'(meri|mere|my)\s*booking\s*(ka|ki|kya|status)', 'booking_status'),
        (r'status\s*of\s*(my\s*)?booking', 'booking_status'),
        (r'booking\s*(confirm|ho\s*gayi?|hua|done)', 'booking_status'),
        
        # Penalty/Fine Reason
        (r'(fine|penalty|jurmana|challan)\s*(reason|kyu|why|explain|wajah)', 'penalty_reason'),
        (r'(fine|penalty)\s*(ka|ki|ke)\s*(reason|wajah|karan)', 'penalty_reason'),
        (r'(fine|penalty)\s*(on|mere|my|laga)', 'penalty_reason'),
        (r'explain\s*(the\s*)?(fine|penalty)', 'penalty_reason'),
        (r'kyu\s*(fine|penalty|challan)\s*laga', 'penalty_reason'),
        
        # Onboarding Status (registration check)
        (r'(registration|account|application)\s*(status|approve|confirm|hua)', 'driver_onboarding_status'),
        (r'mera\s*(registration|account)\s*(hua|ho\s*gaya|approve)', 'driver_onboarding_status'),
        (r'(kya|has|is)\s*(mera|my)\s*account\s*(approve|active)', 'driver_onboarding_status'),
        (r'(account|registration)\s*(ho\s*gaya|approve\s*hua)', 'driver_onboarding_status'),
        
        # ===== AGENT HANDOFF PATTERNS =====
        
        # Booking Cancel
        (r'booking\s*(cancel|raddh?|band)', 'booking_cancel'),
        (r'cancel\s*(my\s*)?booking', 'booking_cancel'),
        (r'(meri|my)\s*booking\s*(cancel|raddh)', 'booking_cancel'),
        (r'booking\s*(ko\s*)?(cancel|hatao)', 'booking_cancel'),
        
        # Driver Onboarding (new registration)
        (r'(driver|rider)\s*(ban|join|register|banna)', 'driver_onboarding'),
        (r'(join|register)\s*(karna|kaise|chahta)', 'driver_onboarding'),
        (r'battery\s*smart\s*(join|mein|me)', 'driver_onboarding'),
        (r'naya\s*(driver|account)\s*(banana|register)', 'driver_onboarding'),
        (r'(driver|rider)\s*kaise\s*ban[eu]?', 'driver_onboarding'),
        
        # Driver Deboarding (account closure)
        (r'(account|driver)\s*(close|band|delete|remove|hatao)', 'driver_deboarding'),
        (r'deboard(ing)?', 'driver_deboarding'),
        (r'account\s*(close|band)\s*karna', 'driver_deboarding'),
        (r'(chhodna|leave|quit)\s*(driver|account|service)', 'driver_deboarding'),
    ]
    
    # Entity extraction patterns
    ENTITY_PATTERNS = {
        'driver_id': r'(?:driver\s*(?:id)?|AB)\s*([A-Z]{2}\d{6}|\d{6})',
        'booking_id': r'(?:booking\s*(?:id|no|number)?|#)\s*(\d{7,8})',
        'phone': r'(?:no|number|phone)?\s*(\d{10,11})',
    }
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the Tier 1 classifier
        
        Args:
            model_path: Path to trained ML model (optional)
            confidence_threshold: Minimum confidence for ML predictions
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tokenizer = None
        self.id2label = None
        
        if model_path and ML_AVAILABLE:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load ML model"""
        try:
            from pathlib import Path
            import json
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            
            label_path = Path(model_path) / "label_mapping.json"
            with open(label_path, 'r') as f:
                label_mapping = json.load(f)
                self.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
            
            print(f"✅ ML model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load ML model: {e}")
            self.model = None
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract entities from text"""
        entities = {}
        text_lower = text.lower()
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities[entity_type] = match.group(1)
        
        return entities
    
    def _apply_rules(self, text: str) -> Tuple[Optional[str], float]:
        """Apply rule-based pattern matching"""
        text_lower = text.lower().strip()
        
        for pattern, intent in self.PATTERN_RULES:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return intent, 1.0
        
        return None, 0.0
    
    def _predict_ml(self, text: str) -> Tuple[str, float, List]:
        """Use ML model for prediction"""
        if not self.model:
            return None, 0.0, []
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        
        pred_id = torch.argmax(probs).item()
        confidence = probs[0][pred_id].item()
        intent = self.id2label[pred_id]
        
        # Get top 3
        top_probs, top_indices = torch.topk(probs[0], min(3, len(probs[0])))
        top_3 = [
            {"intent": self.id2label[idx.item()], "confidence": round(prob.item(), 3)}
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return intent, confidence, top_3
    
    def classify(self, text: str) -> IntentResult:
        """
        Main classification method
        
        Returns IntentResult with:
        - intent: Detected intent
        - confidence: Confidence score (0-1)
        - category: TIER_1 or AGENT
        - method: 'rule', 'model', or 'fallback'
        - action: What to do next
        """
        # Extract entities
        entities = self._extract_entities(text)
        
        # Step 1: Try rule-based matching first (fast, 100% confident)
        rule_intent, rule_conf = self._apply_rules(text)
        
        if rule_intent and rule_intent in self.INTENT_CONFIG:
            config = self.INTENT_CONFIG[rule_intent]
            return IntentResult(
                intent=rule_intent,
                confidence=rule_conf,
                category=config['category'],
                method='rule',
                action=config['action'],
                entities=entities,
                top_predictions=[{'intent': rule_intent, 'confidence': 1.0}]
            )
        
        # Step 2: Try ML model
        if self.model:
            ml_intent, ml_conf, top_preds = self._predict_ml(text)
            
            if ml_conf >= self.confidence_threshold and ml_intent in self.INTENT_CONFIG:
                config = self.INTENT_CONFIG[ml_intent]
                return IntentResult(
                    intent=ml_intent,
                    confidence=ml_conf,
                    category=config['category'],
                    method='model',
                    action=config['action'],
                    entities=entities,
                    top_predictions=top_preds
                )
            
            # Low confidence - handoff to agent
            return IntentResult(
                intent='unknown',
                confidence=ml_conf,
                category=IntentCategory.AGENT,
                method='fallback',
                action='Transfer to agent (low confidence)',
                entities=entities,
                top_predictions=top_preds
            )
        
        # Step 3: No model, no rule match - handoff to agent with helpful message
        return IntentResult(
            intent='unknown',
            confidence=0.0,
            category=IntentCategory.AGENT,
            method='fallback',
            action='Query not recognized - Transfer to agent. This system handles: wallet balance, swap station info, booking status, swap history, swap process, penalties, registration status, booking cancel, driver onboarding/deboarding.',
            entities=entities,
            top_predictions=[]
        )
    
    def predict(self, text: str) -> Dict:
        """Alias for classify() that returns dict"""
        return self.classify(text).to_dict()


# ==================== RESPONSE TEMPLATES ====================

TIER1_RESPONSES = {
    'wallet_balance': {
        'hindi': 'आपके वॉलेट में ₹{balance} बैलेंस है।',
        'english': 'Your wallet balance is ₹{balance}.',
        'hinglish': 'Aapke wallet mein ₹{balance} balance hai.'
    },
    'battery_swap_status': {
        'hindi': 'सबसे नजदीकी स्वैप स्टेशन {station_name} है, {address} पर।',
        'english': 'Nearest swap station is {station_name} at {address}.',
        'hinglish': 'Sabse pass swap station {station_name} hai, {address} pe.'
    },
    'booking_status': {
        'hindi': 'आपकी बुकिंग #{booking_id} की स्थिति: {status}',
        'english': 'Your booking #{booking_id} status: {status}',
        'hinglish': 'Aapki booking #{booking_id} ka status: {status}'
    },
    'swap_history': {
        'hindi': 'आपने पिछले महीने {count} बार बैटरी स्वैप किया।',
        'english': 'You have done {count} battery swaps last month.',
        'hinglish': 'Aapne last month {count} baar battery swap kiya.'
    },
    'swap_process': {
        'hindi': 'बैटरी स्वैप के लिए:\n1. स्टेशन पर जाएं\n2. QR कोड स्कैन करें\n3. पुरानी बैटरी निकालें\n4. नई बैटरी लगाएं',
        'english': 'To swap battery:\n1. Go to station\n2. Scan QR code\n3. Remove old battery\n4. Insert new battery',
        'hinglish': 'Battery swap ke liye:\n1. Station pe jao\n2. QR code scan karo\n3. Purani battery nikalo\n4. Nayi battery lagao'
    },
    'penalty_reason': {
        'hindi': 'आपके अकाउंट पर ₹{amount} का फाइन है। कारण: {reason}',
        'english': 'Your account has a fine of ₹{amount}. Reason: {reason}',
        'hinglish': 'Aapke account pe ₹{amount} ka fine hai. Reason: {reason}'
    },
    'driver_onboarding_status': {
        'hindi': 'आपका रजिस्ट्रेशन {status}। {message}',
        'english': 'Your registration is {status}. {message}',
        'hinglish': 'Aapka registration {status} hai. {message}'
    }
}

AGENT_HANDOFF_MESSAGES = {
    'booking_cancel': 'आपकी बुकिंग कैंसल करने के लिए मैं आपको हमारे एजेंट से कनेक्ट कर रहा हूं।',
    'driver_onboarding': 'नए ड्राइवर रजिस्ट्रेशन के लिए मैं आपको एजेंट से जोड़ रहा हूं।',
    'driver_deboarding': 'अकाउंट बंद करने के लिए कृपया एजेंट से बात करें। कनेक्ट कर रहा हूं...',
    'unknown': 'माफ़ कीजिए, मैं समझ नहीं पाया। आपको एजेंट से कनेक्ट कर रहा हूं।'
}


# ==================== TEST FUNCTION ====================

def test_classifier():
    """Test the classifier with sample queries"""
    
    # Initialize without ML model (rule-based only)
    classifier = Tier1Classifier()
    
    test_cases = [
        # TIER 1 - Wallet
        ("wallet balance check karo", "wallet_balance", "tier1"),
        ("mere wallet mein kitna paisa hai", "wallet_balance", "tier1"),
        ("wallet status for driver AB123456", "wallet_balance", "tier1"),
        
        # TIER 1 - Swap Station
        ("swap station kaha hai", "battery_swap_status", "tier1"),
        ("nearest swap station", "battery_swap_status", "tier1"),
        
        # TIER 1 - Swap History
        ("swap history dikhao", "swap_history", "tier1"),
        ("mera swap record", "swap_history", "tier1"),
        
        # TIER 1 - Swap Process
        ("battery kaise swap kare", "swap_process", "tier1"),
        ("swap process batao", "swap_process", "tier1"),
        
        # TIER 1 - Booking Status
        ("meri booking ka status", "booking_status", "tier1"),
        ("booking confirm hui #12345678", "booking_status", "tier1"),
        
        # TIER 1 - Penalty
        ("fine kyu laga", "penalty_reason", "tier1"),
        ("penalty reason batao", "penalty_reason", "tier1"),
        
        # TIER 1 - Onboarding Status
        ("mera registration status", "driver_onboarding_status", "tier1"),
        ("account approve hua kya", "driver_onboarding_status", "tier1"),
        
        # AGENT - Booking Cancel
        ("booking cancel kardo", "booking_cancel", "agent"),
        ("cancel my booking #87654321", "booking_cancel", "agent"),
        
        # AGENT - Onboarding
        ("driver kaise bane", "driver_onboarding", "agent"),
        ("Battery Smart join karna hai", "driver_onboarding", "agent"),
        
        # AGENT - Deboarding
        ("account close karna hai", "driver_deboarding", "agent"),
        ("driver account band karo", "driver_deboarding", "agent"),
    ]
    
    print("\n" + "=" * 80)
    print("🧪 TIER 1 CLASSIFIER TEST")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for query, expected_intent, expected_category in test_cases:
        result = classifier.classify(query)
        
        intent_match = result.intent == expected_intent
        category_match = result.category.value == expected_category
        
        if intent_match and category_match:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        print(f"\n{status} Query: '{query}'")
        print(f"   Expected: {expected_intent} ({expected_category})")
        print(f"   Got:      {result.intent} ({result.category.value})")
        if result.entities:
            print(f"   Entities: {result.entities}")
    
    print("\n" + "=" * 80)
    print(f"📊 Results: {passed} passed, {failed} failed ({passed/(passed+failed)*100:.0f}% accuracy)")
    print("=" * 80)
    
    return passed, failed


if __name__ == "__main__":
    test_classifier()
