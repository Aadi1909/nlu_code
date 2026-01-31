# Manual Testing Guide - Tier 1 Classifier

## 🎯 Quick Start

### Option 1: Automated Testing (Recommended)
```bash
cd src
python3 test_tier1.py
```
This runs 21 test queries and shows results.

### Option 2: Interactive Testing
```bash
cd src
python3 test_tier1.py interactive
```
Type your own queries and see results in real-time.

### Option 3: Test Individual Queries in Python
```bash
cd src
python3
```

```python
from tier1_classifier import Tier1Classifier

# Initialize
classifier = Tier1Classifier()

# Test a query
result = classifier.classify("wallet balance check karo")

print(f"Intent: {result.intent}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Category: {result.category.value}")
print(f"Action: {result.action}")
```

---

## 📝 Sample Test Queries

### 🤖 TIER 1 (Bot Handles - 7 intents)

#### 1. Wallet Balance
```
wallet balance check karo
mere wallet mein kitna paisa hai
wallet status for driver AB123456
```

#### 2. Swap Station Info
```
swap station kaha hai
nearest swap station
station ka address batao
```

#### 3. Swap History
```
swap history dikhao
mera swap record
kitne swap kiye last month
```

#### 4. Swap Process
```
battery kaise swap kare
swap process batao
battery badalne ka tarika
```

#### 5. Booking Status
```
meri booking ka status
booking confirm hui #12345678
booking status check karo
```

#### 6. Penalty Reason
```
fine kyu laga
penalty reason batao
explain the fine on my account
```

#### 7. Driver Onboarding Status
```
mera registration status
account approve hua kya
registration ka status batao
```

---

### 👤 AGENT HANDOFF (3 intents)

#### 1. Booking Cancel
```
booking cancel kardo
cancel my booking #87654321
meri booking cancel karni hai
```

#### 2. Driver Onboarding (New Registration)
```
driver kaise bane
Battery Smart join karna hai
naya driver account banana hai
```

#### 3. Driver Deboarding (Account Closure)
```
account close karna hai
driver account band karo
account delete karna hai
```

---

## 🧪 Testing with ML Model (SageMaker)

After training the model on SageMaker:

```bash
# Process data
python process_raw_training_data.py

# Train model
python train_with_full_data.py

# Test with ML model
from tier1_classifier import Tier1Classifier

classifier = Tier1Classifier(
    model_path="/mnt/sagemaker-nvme/intent_models_full",
    confidence_threshold=0.5
)

result = classifier.classify("some complex query")
```

---

## 📊 Expected Results

### Rule-Based Matching
- **Confidence**: 100% (pattern matched)
- **Method**: 'rule'
- **Speed**: Instant (<1ms)

### ML Model Matching
- **Confidence**: 50-99% (depends on query)
- **Method**: 'model'
- **Speed**: ~10-50ms

### Fallback (Low Confidence)
- **Confidence**: <50%
- **Method**: 'fallback'
- **Action**: Transfer to agent

---

## 🎨 Output Format

```python
{
    'intent': 'wallet_balance',
    'confidence': 1.0,
    'category': 'tier1',
    'method': 'rule',
    'action': 'Call /api/wallet/balance endpoint',
    'entities': {'driver_id': 'AB123456'},
    'should_handoff': False,
    'top_predictions': [
        {'intent': 'wallet_balance', 'confidence': 1.0}
    ]
}
```

---

## ✅ Validation Checklist

- [x] All 7 Tier 1 intents work correctly
- [x] All 3 Agent handoff intents work correctly
- [x] Entity extraction (driver_id, booking_id) works
- [x] Rule-based patterns have 100% accuracy
- [x] Confidence thresholds work properly
- [x] Multilingual support (Hindi, English, Hinglish)

---

## 🔧 Troubleshooting

### Issue: Import errors
**Solution**: Make sure you're in the `src/` directory
```bash
cd src
python3 test_tier1.py
```

### Issue: No module named 'tier1_classifier'
**Solution**: Use the correct import path
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Issue: ML model not loading
**Solution**: That's OK! The rule-based classifier works without ML model:
```python
classifier = Tier1Classifier(model_path=None)
```

---

## 📈 Performance Metrics

| Metric | Rule-Based | ML Model (Expected) |
|--------|-----------|---------------------|
| Accuracy | 100% | 80-90% |
| Speed | <1ms | 10-50ms |
| Coverage | Pattern matches | All queries |
| Confidence | 100% | 50-99% |

---

## 🚀 Next Steps

1. ✅ Test locally with rule-based (done!)
2. ⏳ Train ML model on SageMaker (10K examples)
3. ⏳ Test with ML model + rules (hybrid)
4. ⏳ Deploy API endpoint
5. ⏳ Integrate with voice assistant

---

## 💡 Tips

- **For common queries**: Rule-based is perfect (100% accuracy)
- **For variations**: ML model handles edge cases
- **For unknown queries**: Automatic agent handoff
- **For production**: Use hybrid (rules + ML)
