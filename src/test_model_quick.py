import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Load model from NVMe disk
model_path = "/mnt/sagemaker-nvme/intent_models"
print(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label mapping
with open(f"{model_path}/label_mapping.json", 'r') as f:
    label_mapping = json.load(f)
    id2label = {int(k): v for k, v in label_mapping["id2label"].items()}

print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)

# Load and display classification report
with open(f"{model_path}/classification_report.json", 'r') as f:
    report = json.load(f)
    
print(f"\n📊 Overall Metrics:")
print(f"   Accuracy:  {report['accuracy']:.2%}")
print(f"   Precision: {report['weighted avg']['precision']:.2%}")
print(f"   Recall:    {report['weighted avg']['recall']:.2%}")
print(f"   F1-Score:  {report['weighted avg']['f1-score']:.2%}")

print(f"\n🎯 Top Performing Intents (by F1-score):")
intents_f1 = [(k, v['f1-score'], v['support']) for k, v in report.items() 
              if isinstance(v, dict) and 'f1-score' in v and v['support'] > 0]
intents_f1.sort(key=lambda x: x[1], reverse=True)

for intent, f1, support in intents_f1[:5]:
    print(f"   {intent:30s} - F1: {f1:.2%} (samples: {int(support)})")

print(f"\n⚠️  Intents Needing Improvement (F1 < 0.3):")
for intent, f1, support in intents_f1:
    if f1 < 0.3:
        print(f"   {intent:30s} - F1: {f1:.2%} (samples: {int(support)})")

# Test with sample queries
print("\n" + "=" * 70)
print("TESTING WITH SAMPLE QUERIES")
print("=" * 70)

test_queries = [
    "मुझे अपना बैलेंस चेक करना है",
    "I want to check my subscription",
    "battery kitna hai?",
    "swap station kaha hai?",
    "मुझे payment करनी है",
]

for query in test_queries:
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    
    pred_id = torch.argmax(probs).item()
    confidence = probs[0][pred_id].item()
    intent = id2label[pred_id]
    
    print(f"\n📝 Query: '{query}'")
    print(f"   ✅ Intent: {intent}")
    print(f"   🎲 Confidence: {confidence:.2%}")
    
    # Show top 3
    top_probs, top_indices = torch.topk(probs[0], 3)
    print(f"   Top 3:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        print(f"      {i}. {id2label[idx.item()]:25s} ({prob.item():.1%})")

print("\n" + "=" * 70)
print("✅ Testing Complete!")
print("=" * 70)
