#!/usr/bin/env python3
"""
Advanced data augmentation to generate diverse training examples
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import Counter

class AdvancedDataAugmenter:
    """Generate diverse training data"""
    
    # Synonym replacements for better diversity
    SYNONYMS = {
        # Battery related
        'battery': ['battery', 'baitri', 'charge', 'cell', 'power'],
        'check': ['check', 'dekho', 'dekhna', 'batao', 'bataiye', 'pata karo', 'verify'],
        'status': ['status', 'sthiti', 'condition', 'halat', 'kya hai'],
        'range': ['range', 'distance', 'kitne kilometer', 'kitna door', 'how far'],
        
        # Swap related
        'swap': ['swap', 'change', 'badalna', 'exchange', 'replace', 'badlo'],
        'station': ['station', 'center', 'kendra', 'point', 'location', 'jagah'],
        'nearest': ['nearest', 'najdik', 'paas', 'closest', 'sabse paas'],
        'where': ['where', 'kaha', 'kidhar', 'location', 'address'],
        'direction': ['direction', 'rasta', 'route', 'path', 'kaise jaye'],
        
        # Subscription related
        'subscription': ['subscription', 'plan', 'package', 'membership', 'yojana', 'scheme'],
        'renew': ['renew', 'renewal', 'badalna', 'extend', 'continue'],
        'upgrade': ['upgrade', 'better', 'improve', 'badiya', 'enhance'],
        
        # Payment related
        'payment': ['payment', 'pay', 'paise', 'paisa', 'bhugtan', 'money'],
        'make': ['make', 'karna', 'do', 'complete', 'process'],
        'history': ['history', 'pichle', 'previous', 'past', 'old'],
        'pending': ['pending', 'due', 'baaki', 'remaining', 'outstanding'],
        
        # Help/Agent related
        'help': ['help', 'madad', 'sahayata', 'assist', 'support'],
        'problem': ['problem', 'issue', 'dikkat', 'samasya', 'trouble'],
        'agent': ['agent', 'representative', 'person', 'customer care', 'support team'],
        'talk': ['talk', 'speak', 'baat', 'connect', 'contact'],
        
        # Greetings
        'hi': ['hi', 'hello', 'hey', 'namaste', 'namaskar', 'hii'],
        'bye': ['bye', 'goodbye', 'tata', 'alvida', 'see you'],
        'thanks': ['thanks', 'thank you', 'dhanyavaad', 'shukriya'],
    }
    
    # Template patterns for each intent
    INTENT_TEMPLATES = {
        'check_battery_status': [
            "मेरी battery का status क्या है?",
            "battery कितनी है?",
            "मुझे battery check करना है",
            "battery level बताओ",
            "मेरी bike की battery kitna hai?",
            "baitri कितनी charge hai?",
            "can you check my battery?",
            "what is my battery status?",
            "tell me battery percentage",
            "how much battery do I have?",
        ],
        
        'check_battery_range': [
            "battery से कितने kilometer चल सकता है?",
            "battery range क्या है?",
            "kitne km tak chalegi battery?",
            "मेरी battery से कितना door जा सकते हैं?",
            "battery range check karo",
            "how far can I go with current battery?",
            "what is my battery range?",
            "how many kilometers can I travel?",
            "battery distance kitna hai?",
        ],
        
        'find_swap_station': [
            "swap station kaha hai?",
            "nearest battery swap station बताओ",
            "मुझे swap karna hai, kendra kaha hai?",
            "battery change karne ke liye station kaha hai?",
            "najdik ka swap center kaha hai?",
            "where is the nearest swap station?",
            "find swap station near me",
            "battery kendra ka location batao",
            "swap point kaha milega?",
        ],
        
        'station_directions': [
            "swap station ka rasta batao",
            "station tak kaise jaye?",
            "swap center ka direction chahiye",
            "kendra tak pohchne ke liye route batao",
            "how to reach swap station?",
            "give me directions to station",
            "station tak pohchane ka raasta",
        ],
        
        'check_station_availability': [
            "station pe battery available hai kya?",
            "swap station open hai?",
            "battery milegi station pe?",
            "is the station operational?",
            "station me battery hai kya?",
            "kendra me battery stock hai?",
        ],
        
        'check_subscription': [
            "मेरा subscription क्या है?",
            "मेरी plan kya hai?",
            "subscription status बताओ",
            "what is my current plan?",
            "my membership details chahiye",
            "mera package kya hai?",
        ],
        
        'subscription_renewal': [
            "subscription renew karna hai",
            "plan ko renew karo",
            "मुझे plan extend करना है",
            "subscription badalna hai",
            "I want to renew my plan",
            "yojana ko renew kaise kare?",
        ],
        
        'upgrade_subscription': [
            "better plan chahiye",
            "subscription upgrade karo",
            "मुझे better package chahiye",
            "upgrade my plan",
            "मुझे higher plan me shift hona hai",
        ],
        
        'swap_count': [
            "kitne swap kiye maine?",
            "मेरे swap count बताओ",
            "how many times did I swap?",
            "maine kitni bar battery badli?",
            "swap history count chahiye",
        ],
        
        'swap_history': [
            "मेरी swap history batao",
            "pichle swaps dekhna hai",
            "previous battery changes बताओ",
            "swap records chahiye",
            "last swaps ka data batao",
        ],
        
        'check_ride_stats': [
            "मेरी ride statistics बताओ",
            "kitne kilometer chala?",
            "मेरे riding data chahiye",
            "trip history dekho",
            "total distance traveled",
        ],
        
        'make_payment': [
            "payment karna hai",
            "मुझे paise pay karne hai",
            "I want to make payment",
            "bhugtan karna hai",
            "bill payment karo",
        ],
        
        'payment_history': [
            "payment history batao",
            "pichle payments dekho",
            "मेरे past payments chahiye",
            "previous bill records",
            "purane payments ka data",
        ],
        
        'payment_inquiry': [
            "payment pending hai kya?",
            "kitna paisa baaki hai?",
            "due amount बताओ",
            "outstanding payment kitna hai?",
            "mera bill kitna hai?",
        ],
        
        'report_issue': [
            "मुझे problem hai",
            "bike me dikkat hai",
            "issue report karna hai",
            "complaint karna chahta hu",
            "something is wrong",
            "samasya hai mere bike me",
        ],
        
        'help_general': [
            "मुझे help chahiye",
            "मुझे kuch batao",
            "I need assistance",
            "kuch samajh nahi aa raha",
            "guide me please",
        ],
        
        'talk_to_agent': [
            "agent se baat karni hai",
            "customer care se connect karo",
            "insaan se baat karna hai",
            "I want to talk to someone",
            "representative se baat chahiye",
        ],
        
        'greet': [
            "hi", "hello", "hey", "namaste", "namaskar",
            "good morning", "good evening", "hii", "helo",
        ],
        
        'bye': [
            "bye", "goodbye", "tata bye", "alvida",
            "see you", "good bye", "thik hai bye",
        ],
        
        'thank': [
            "thanks", "thank you", "dhanyavaad", "shukriya",
            "thanks a lot", "bahut dhanyavaad", "appreciate it",
        ],
        
        'affirm': [
            "yes", "haan", "okay", "ok", "sure", "thik hai",
            "bilkul", "zaroor", "right", "correct",
        ],
        
        'deny': [
            "no", "nahi", "nope", "no thanks", "mana",
            "galat", "incorrect", "nahi chahiye",
        ],
    }
    
    def augment_by_synonym_replacement(self, text: str, num_aug: int = 2) -> List[str]:
        """Replace words with synonyms"""
        augmented = []
        words = text.lower().split()
        
        for _ in range(num_aug):
            new_words = words.copy()
            for i, word in enumerate(words):
                if word in self.SYNONYMS and random.random() > 0.5:
                    new_words[i] = random.choice(self.SYNONYMS[word])
            
            aug_text = ' '.join(new_words)
            if aug_text != text.lower():
                augmented.append(aug_text)
        
        return augmented
    
    def generate_from_templates(self, target_samples: int = 100) -> List[Dict]:
        """Generate training data from templates"""
        data = []
        
        for intent, templates in self.INTENT_TEMPLATES.items():
            print(f"Generating for {intent}...")
            
            # Add all templates
            for template in templates:
                data.append({
                    'text': template,
                    'intent': intent,
                    'language': 'mixed'
                })
            
            # Augment templates
            while len([d for d in data if d['intent'] == intent]) < target_samples:
                template = random.choice(templates)
                augmented = self.augment_by_synonym_replacement(template, num_aug=1)
                
                for aug_text in augmented:
                    if len([d for d in data if d['intent'] == intent]) >= target_samples:
                        break
                    
                    data.append({
                        'text': aug_text,
                        'intent': intent,
                        'language': 'mixed'
                    })
        
        return data
    
    def merge_with_existing(self, new_data: List[Dict]) -> List[Dict]:
        """Merge with existing balanced data"""
        existing_path = Path("../data/processed/train_balanced.json")
        
        if existing_path.exists():
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            print(f"\nMerging with existing {len(existing_data)} samples...")
            
            # Combine and deduplicate
            all_data = existing_data + new_data
            
            # Remove duplicates based on text
            seen_texts = set()
            deduplicated = []
            for item in all_data:
                text_lower = item['text'].lower().strip()
                if text_lower not in seen_texts:
                    seen_texts.add(text_lower)
                    deduplicated.append(item)
            
            return deduplicated
        
        return new_data


def main():
    print("🚀 Starting Advanced Data Augmentation...\n")
    
    augmenter = AdvancedDataAugmenter()
    
    # Generate diverse data
    print("=" * 70)
    print("GENERATING DIVERSE TRAINING DATA")
    print("=" * 70)
    
    new_data = augmenter.generate_from_templates(target_samples=100)
    print(f"\n✅ Generated {len(new_data)} new samples")
    
    # Merge with existing
    final_data = augmenter.merge_with_existing(new_data)
    
    # Add labels
    intent_to_id = {}
    for item in final_data:
        if item['intent'] not in intent_to_id:
            intent_to_id[item['intent']] = len(intent_to_id)
        item['label'] = intent_to_id[item['intent']]
    
    # Save
    output_path = Path("../data/processed/train_augmented_v2.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved augmented dataset to: {output_path}")
    print(f"   Total samples: {len(final_data)}")
    
    # Show distribution
    intent_counts = Counter([item['intent'] for item in final_data])
    print(f"\n📊 Final distribution (top 10):")
    for intent, count in intent_counts.most_common(10):
        print(f"   {intent:30s}: {count:3d} samples")
    
    print(f"\n📊 Summary:")
    print(f"   Total intents: {len(intent_counts)}")
    print(f"   Average samples per intent: {len(final_data) / len(intent_counts):.0f}")
    print(f"   Min samples: {min(intent_counts.values())}")
    print(f"   Max samples: {max(intent_counts.values())}")
    
    print("\n" + "=" * 70)
    print("✅ DATA AUGMENTATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Train with improved data")
    print("  python train_intent_classifier_improved.py")


if __name__ == "__main__":
    main()
