# Quick Start Guide - Battery Smart NLU New Features

## 🚀 Quick Overview

You've successfully added **20 new intents** covering the complete Battery Smart driver journey!

### What Was Added:
- ✅ Driver onboarding (4 intents)
- ✅ Vehicle retrofitment (2 intents)  
- ✅ Pricing & subscriptions (3 intents)
- ✅ DSK operations (3 intents)
- ✅ Driver status management (2 intents)
- ✅ Inactivity Center (2 intents)
- ✅ Partner stations (3 intents)
- ✅ Benefits info (1 intent)

---

## 📁 Files Created/Modified

### Created:
1. `data/raw/new_intents_training_data.json` - 82 training examples
2. `merge_training_data.py` - Data merging script
3. `NEW_FEATURES_DOCUMENTATION.md` - Complete documentation
4. `API_INTEGRATION_GUIDE.md` - API specs for backend team
5. `IMPLEMENTATION_SUMMARY.md` - This summary
6. `QUICK_START_GUIDE.md` - You are here!

### Modified:
1. `config/intents.yaml` - Added 20 intents with 200+ examples
2. `config/entities.yaml` - Added 13 entities
3. `config/slots.yml` - Added slot mappings
4. `config/responses.yaml` - Added 14 response templates

---

## ⚡ Quick Test

### 1. Verify Configuration
```bash
cd /Users/divyansh.masand/Desktop/model_proj

# Check intents file
grep -c "intent:" config/intents.yaml
# Should show 29 intents

# Check training data
python3 -c "import json; print(len(json.load(open('data/raw/training_data.json'))))"
# Should show 10082
```

### 2. Test Queries (Manual)

Try these in your test interface:

**Hindi:**
```
Battery Smart join karna hai
meri onboarding kahan tak hui
retrofitment kab hogi
swap ki price kya hai
DSK par onboard karna hai
```

**English:**
```
I want to become a driver
Check my onboarding status
Schedule kit installation
What is swap price
Register at Driver Seva Kendra
```

**Hinglish:**
```
driver banna hai
verification status check karo
kit lagane ka appointment book karo
partner station kahan hai
battery submit karni hai IC par
```

---

## 🎯 Next Steps

### Immediate (Do Now):

1. **Review Configuration**
   ```bash
   # Check new intents
   cat config/intents.yaml | grep "driver_onboarding" -A 20
   
   # Check new entities
   cat config/entities.yaml | grep "phone_number" -A 10
   ```

2. **Verify Training Data Merge**
   ```bash
   # Check if backup exists
   ls -lh data/raw/training_data.json.backup
   
   # Verify new intent distribution
   python3 -c "
   import json
   from collections import Counter
   data = json.load(open('data/raw/training_data.json'))
   intents = Counter(item['intent'] for item in data)
   print(f'Total examples: {len(data)}')
   print(f'Unique intents: {len(intents)}')
   print('\\nNew intents:')
   for intent in sorted(intents.keys()):
       if intent in ['driver_onboarding', 'dsk_onboard', 'ic_battery_submit', 'partner_station_location']:
           print(f'  {intent}: {intents[intent]}')
   "
   ```

### Short-term (This Week):

3. **Retrain Models**
   ```bash
   # Option 1: Train specific models
   python3 src/training/train_intent.py
   python3 src/training/train_entity.py
   
   # Option 2: Train all models
   python3 src/training/train_all_models.py
   ```

4. **Test NLU Pipeline**
   ```bash
   # Run automated tests
   python3 src/nlu/test_nlu_pipeline.py
   
   # Manual testing
   python3 src/test_manual.py
   ```

5. **Validate Responses**
   ```bash
   # Check response templates load correctly
   python3 -c "
   import yaml
   responses = yaml.safe_load(open('config/responses.yaml'))
   new_responses = ['driver_onboarding_response', 'dsk_onboard_response', 'ic_battery_submit_response']
   for resp in new_responses:
       if resp in responses:
           print(f'✓ {resp} loaded')
       else:
           print(f'✗ {resp} missing')
   "
   ```

### Medium-term (This Month):

6. **API Integration**
   - Share `API_INTEGRATION_GUIDE.md` with backend team
   - Implement all 18 new API endpoints
   - Test API responses match expected format

7. **Load Testing**
   - Test with high volume of queries
   - Measure intent classification accuracy
   - Measure entity extraction accuracy

8. **User Testing**
   - Test with real drivers
   - Collect feedback on responses
   - Iterate on prompts and examples

---

## 📊 Expected Metrics

After retraining, you should see:

### Intent Classification:
- **Accuracy**: >95% for new intents
- **Confusion**: <2% between similar intents (e.g., dsk_onboard vs driver_deactivate)

### Entity Extraction:
- **Phone numbers**: >98% accuracy
- **Dates/times**: >95% accuracy
- **Location IDs**: >90% accuracy

### Response Quality:
- **Language detection**: >99% accuracy
- **Template selection**: 100% (rule-based)
- **Slot filling**: >90% auto-fill success

---

## 🔍 Troubleshooting

### Issue: Training data not merged
```bash
# Re-run merge script
python3 merge_training_data.py

# Check if backup exists (means merge succeeded)
ls data/raw/training_data.json.backup
```

### Issue: Intent not recognized
```bash
# Check if intent exists in config
grep "intent: driver_onboarding" config/intents.yaml

# Check training examples
python3 -c "
import json
data = json.load(open('data/raw/training_data.json'))
examples = [item for item in data if item['intent'] == 'driver_onboarding']
print(f'Found {len(examples)} examples for driver_onboarding')
for ex in examples[:3]:
    print(f\"  - {ex['text']}\")
"
```

### Issue: Entity not extracted
```bash
# Check entity definition
grep "entity: phone_number" -A 10 config/entities.yaml

# Test regex pattern
python3 -c "
import re
pattern = r'[6-9]\d{9}'
test_numbers = ['9876543210', '1234567890', '8888888888']
for num in test_numbers:
    match = re.match(pattern, num)
    print(f'{num}: {\"✓\" if match else \"✗\"}')
"
```

### Issue: Response in wrong language
```bash
# Check response template
grep "driver_onboarding_response" -A 20 config/responses.yaml

# Verify all 3 languages present
python3 -c "
import yaml
responses = yaml.safe_load(open('config/responses.yaml'))
template = responses['driver_onboarding_response']['templates']
for lang in ['en', 'hi', 'hinglish']:
    if lang in template:
        print(f'✓ {lang} template exists')
    else:
        print(f'✗ {lang} template missing')
"
```

---

## 📚 Documentation Index

1. **NEW_FEATURES_DOCUMENTATION.md** - Complete feature descriptions with examples
2. **API_INTEGRATION_GUIDE.md** - API specs for backend integration
3. **IMPLEMENTATION_SUMMARY.md** - What was implemented and statistics
4. **QUICK_START_GUIDE.md** - This file (getting started)

---

## 💡 Pro Tips

### Tip 1: Test Incrementally
Don't test all 20 intents at once. Group by category:
- Day 1: Onboarding (4 intents)
- Day 2: Retrofitment + Pricing (5 intents)
- Day 3: DSK + Driver Status (5 intents)
- Day 4: IC + Partner Stations (6 intents)

### Tip 2: Use Real Scenarios
Test with actual user journeys:
```
User Journey 1: New Driver
1. "Battery Smart join karna hai"
2. "onboarding status check karo"
3. "verification kab hoga"
4. "retrofitment schedule karo"

User Journey 2: Going on Leave
1. "DSK kahan hai"
2. "leave leni hai 10 days ki"
3. "DSK par onboard karna hai"

User Journey 3: Daily Swap
1. "partner station kahan hai"
2. "swap kaise hota hai"
3. "battery scan nahi ho rahi"
```

### Tip 3: Monitor Confidence Scores
```python
# In your NLU pipeline
if intent_confidence < 0.85:
    # Ask for clarification
    # Or transfer to agent

if intent_confidence > 0.95:
    # High confidence - proceed
```

### Tip 4: Collect User Feedback
After each interaction:
- "Was this helpful?" (Yes/No)
- If No: "What were you looking for?"
- Use feedback to improve training data

---

## 🎉 Success Checklist

- [ ] All config files validated (no syntax errors)
- [ ] Training data merged successfully (10,082 examples)
- [ ] Backup created (training_data.json.backup exists)
- [ ] New intents appear in training data
- [ ] All 3 languages have response templates
- [ ] Entity patterns tested with sample data
- [ ] Slot mappings defined for all intents
- [ ] API endpoint specifications reviewed
- [ ] Models retrained with new data
- [ ] Manual testing completed for each category
- [ ] API integration guide shared with backend team

---

## 🆘 Need Help?

### Check These First:
1. Error logs in terminal output
2. Configuration file syntax (YAML indentation)
3. Training data format (valid JSON)
4. Entity regex patterns (test with sample data)

### Common Issues:
- **YAML syntax error**: Check indentation (use spaces, not tabs)
- **JSON parse error**: Validate JSON with `python3 -m json.tool file.json`
- **Intent not found**: Check spelling in intents.yaml
- **Entity not extracted**: Test regex pattern separately

---

## 🎓 Learning Resources

### Understanding the Flow:
```
User Input (Voice/Text)
    ↓
Language Detection (Hi/En/Hinglish)
    ↓
Intent Classification (Which intent?)
    ↓
Entity Extraction (Extract data)
    ↓
Slot Filling (Get missing info)
    ↓
API Call (Fetch data)
    ↓
Response Template Selection
    ↓
Response Generation (Fill template)
    ↓
Output to User
```

### Key Components:
1. **Intents**: What user wants (driver_onboarding, dsk_onboard, etc.)
2. **Entities**: Data to extract (phone_number, dsk_location, etc.)
3. **Slots**: Required/optional data for API call
4. **Responses**: What to say back to user

---

## 🚀 Ready to Go!

You now have:
- ✅ 20 new intents configured
- ✅ 13 new entities defined
- ✅ 82+ training examples
- ✅ Multi-language support (3 languages)
- ✅ Complete API specifications
- ✅ Comprehensive documentation

**Next action**: Run the training pipeline and start testing! 🎊

```bash
# Train models
python3 src/training/train_all_models.py

# Start testing
python3 src/test_manual.py
```

---

## 📞 Contact

For questions:
- Check documentation in this repo
- Review training data format
- Test individual components
- Contact: Your friendly AI assistant! 😊

---

**Good luck with your Battery Smart NLU system!** 🚗⚡🔋
