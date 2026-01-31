# 🚀 AWS Bedrock Training Instructions

## ✅ What's Been Done

1. **Data Augmentation Complete**
   - Original: 342 training examples
   - Augmented: 1,230 training examples (3.6x increase)
   - All 23 intents balanced (40-96 examples each)

2. **Code Improvements**
   - Updated hyperparameters for better performance
   - Added data augmentation script
   - Created testing infrastructure
   - Added comprehensive documentation

3. **GitHub Ready**
   - All changes committed and pushed to: `github.com/Aadi1909/nlu_code.git`
   - Branch: `main`
   - Latest commit: "Add AWS Bedrock setup script"

## 📋 Steps to Train on AWS Bedrock

### 1. On AWS Bedrock Instance

```bash
# Clone the repository
git clone https://github.com/Aadi1909/nlu_code.git
cd nlu_code

# Run setup script
chmod +x setup_bedrock.sh
./setup_bedrock.sh

# Start training
cd src
python3 train_intent_classifier.py
```

### 2. Training Details

**Expected Training Time:** 3-5 minutes (on GPU) vs 20+ minutes (on CPU)

**Expected Results:**
- Training accuracy: 85%+
- Validation accuracy: 75%+
- Test accuracy: 70%+
- All 23 intents properly trained

**Training Progress:**
- Total steps: ~1,200 (15 epochs × 77 steps per epoch)
- Evaluation every 50 steps
- Early stopping if no improvement for 5 evaluations

### 3. After Training Completes

The trained model will be saved in:
```
models/intent_classifier/
├── config.json
├── model.safetensors        # Main model file (~360MB)
├── tokenizer_config.json
├── tokenizer.json
├── vocab.txt
├── special_tokens_map.json
├── label_mapping.json       # Intent mappings
└── classification_report.json  # Performance metrics
```

### 4. Commit Trained Model to GitHub

```bash
# Add the trained model
git add models/intent_classifier/

# Commit with performance metrics
git commit -m "Add trained intent classifier (Accuracy: XX.X%)"

# Push to GitHub
git push origin main
```

**Note:** The model files are ~360MB. If Git rejects large files:
```bash
# Install Git LFS (if not already installed)
git lfs install

# Track large model files
git lfs track "models/intent_classifier/*.safetensors"
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"

# Then commit model
git add models/intent_classifier/
git commit -m "Add trained intent classifier"
git push origin main
```

### 5. Pull on Local Machine

```bash
cd /Users/divyansh.masand/Desktop/model_proj
git pull origin main
```

### 6. Test Locally

```bash
cd src
python3 test_intent_model.py
```

## 📊 Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Training Data | 342 | 1,230 |
| Test Accuracy | 48.6% | 70%+ |
| Low-confidence Issues | Many <6% | Most >60% |
| Training Time (GPU) | N/A | 3-5 min |
| Training Time (CPU) | 13 min | 20+ min |

## 🎯 Success Indicators

After training on Bedrock, you should see:

1. **High Accuracy:**
   ```
   Test Results: {'eval_accuracy': 0.75+, 'eval_f1': 0.72+}
   ```

2. **Confident Predictions:**
   ```
   Query: "मेरी बैटरी कितनी बची है?"
   → Intent: check_battery_status (confidence: 85%+)
   ```

3. **All Intents Working:**
   - No intents with 0% recall
   - All classes have F1 > 0.50

## 🔍 Troubleshooting

### If accuracy is still low (<65%):
- Train for more epochs (change `num_epochs=15` to `20`)
- Increase batch size if memory allows
- Check classification report for problematic intents

### If training is slow on Bedrock:
- Verify GPU is available: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Check GPU usage: `nvidia-smi`
- Ensure batch size is appropriate for GPU memory

### If Git push fails (file too large):
- Use Git LFS (see instructions above)
- Or push without model files, train locally after pulling

## 📞 Quick Reference

**Repository:** https://github.com/Aadi1909/nlu_code.git
**Branch:** main
**Training Script:** `src/train_intent_classifier.py`
**Testing Script:** `src/test_intent_model.py`
**Model Output:** `models/intent_classifier/`

---

**Ready to train! 🚀**

Once trained on Bedrock, the model will be production-ready with significantly improved accuracy and confidence scores.
