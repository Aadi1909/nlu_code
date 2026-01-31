# Model Training & Deployment Guide

## Current Status
- ✅ Data augmented: 342 → 1,230 training examples
- ✅ Hyperparameters optimized
- ✅ Dataset balanced across all 23 intents
- 📦 Ready for training on AWS Bedrock

## Files to Push to GitHub

### Training Data (Augmented)
```
data/processed/train_augmented.json
data/processed/val_augmented.json
data/processed/test_augmented.json
```

### Training Scripts
```
src/train_intent_classifier.py (updated with augmented data support)
src/improve_and_retrain.py (data augmentation script)
```

### Configuration
```
config/intents.yaml
config/entities.yaml
config/domain.yaml
requirements.txt
```

## Steps to Train on AWS Bedrock

### 1. Commit and Push Changes
```bash
cd /Users/divyansh.masand/Desktop/model_proj
git add .
git commit -m "Add augmented training data and improved model configuration"
git push origin main
```

### 2. On AWS Bedrock Instance
```bash
# Clone repository
git clone <your-repo-url>
cd model_proj

# Install dependencies
pip install -r requirements.txt

# Run training (with GPU this time!)
cd src
python3 train_intent_classifier.py
```

### 3. After Training, Push Model Back
```bash
# The trained model will be in: models/intent_classifier/
git add models/intent_classifier/
git commit -m "Add trained intent classifier model"
git push origin main
```

### 4. Pull on Local Machine
```bash
git pull origin main
```

### 5. Test Locally
```bash
cd src
python3 test_intent_model.py
```

## Expected Improvements
- **Training Time:** ~3-5 minutes on GPU (vs 20+ on CPU)
- **Accuracy:** 48.6% → 70%+ expected
- **Better Performance:** All 23 intents properly trained

## Model Files to Commit
After training on Bedrock, these files will be created:
```
models/intent_classifier/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── tokenizer.json
├── vocab.txt
├── special_tokens_map.json
├── label_mapping.json
└── classification_report.json
```

## Git Commands Summary
```bash
# Before training on Bedrock
git add data/processed/*_augmented.json
git add src/train_intent_classifier.py
git add src/improve_and_retrain.py
git commit -m "Prepare augmented data for training"
git push

# After training on Bedrock
git add models/intent_classifier/
git commit -m "Add trained model from Bedrock"
git push

# On local machine
git pull
```

## Notes
- Large model files (>100MB) may need Git LFS
- Consider `.gitignore` for cache files
- Augmented data increases repo size by ~2-3MB
