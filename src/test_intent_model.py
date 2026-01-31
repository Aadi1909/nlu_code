#!/usr/bin/env python3
"""Quick script to test the trained intent classifier"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

def load_model(model_path: str):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load label mapping
    with open(Path(model_path) / "label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
    
    return tokenizer, model, id2label

def predict_intent(text: str, tokenizer, model, id2label):
    """Predict intent for given text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    intent = id2label[predicted_class]
    
    return intent, confidence

def main():
    # Load model
    model_path = "../models/intent_classifier"
    tokenizer, model, id2label = load_model(model_path)
    
    print("\n" + "="*60)
    print("Intent Classifier - Interactive Testing")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    # Test examples
    test_queries = [
        "मेरी बैटरी कितनी बची है?",  # How much battery is left?
        "नजदीकी स्वैप स्टेशन कहाँ है?",  # Where is nearest swap station?
        "मुझे एजेंट से बात करनी है",  # I want to talk to agent
        "hi",
        "मेरा सब्सक्रिप्शन कब खत्म होगा?",  # When will my subscription end?
    ]
    
    print("Testing with sample queries:\n")
    for query in test_queries:
        intent, confidence = predict_intent(query, tokenizer, model, id2label)
        print(f"Query: {query}")
        print(f"→ Intent: {intent} (confidence: {confidence:.2%})\n")
    
    # Interactive mode
    print("\n" + "-"*60)
    print("Interactive Mode - Enter your queries:")
    print("-"*60 + "\n")
    
    while True:
        try:
            query = input("Enter query: ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            intent, confidence = predict_intent(query, tokenizer, model, id2label)
            print(f"→ Intent: {intent} (confidence: {confidence:.2%})\n")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye! 👋")

if __name__ == "__main__":
    main()
