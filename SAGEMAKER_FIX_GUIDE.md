# SageMaker Training Fix Guide

## Issues Fixed ✅

### 1. **9,113 Invalid Examples** → Fixed
**Problem**: Validation was checking against hardcoded list of 9 old intents. Your 20 new intents were marked as invalid.

**Solution**: Updated `src/data_preprocessing/data_preparation_pipeline.py` to load valid intents/entities dynamically from `config/intents.yaml` and `config/entities.yaml`.

### 2. **Disk Space Issue** → Solution Below
**Problem**: SageMaker instance ran out of disk space downloading the 1.33GB model.

**Solutions**:
1. **Use NVMe storage** (you're already using it: `/mnt/sagemaker-nvme/`)
2. **Clear HuggingFace cache** before training
3. **Use smaller model** (optional fallback)

---

## Step-by-Step Instructions for SageMaker

### Step 1: Pull Latest Changes
```bash
cd /mnt/sagemaker-nvme/nlu_code
git pull
```

Expected output:
```
Updating 982e5ff7..5598d95d
Fast-forward
 INTENT_API_MAPPING.md                             | 864 +++++++++++++++++++
 data/raw/training_data_cleaned.json               | large file
 diagnose_and_fix_data.py                          | 148 ++++
 src/data_preprocessing/data_preparation_pipeline.py | 78 +-
 4 files changed, 164023 insertions(+), 11 deletions(-)
```

---

### Step 2: Clear HuggingFace Cache (Free Up Space)
```bash
# Remove old cached models
rm -rf /home/sagemaker-user/.cache/huggingface/hub/*

# Check disk space
df -h /home/sagemaker-user/.cache
```

This should free up ~2-3 GB.

---

### Step 3: Set HuggingFace Cache to NVMe (More Space)
```bash
# Set environment variable to use NVMe for cache
export HF_HOME=/mnt/sagemaker-nvme/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/sagemaker-nvme/.cache/huggingface

# Create directory
mkdir -p /mnt/sagemaker-nvme/.cache/huggingface

# Verify
echo $HF_HOME
```

---

### Step 4: Run Data Preprocessing (Should Pass Now)
```bash
cd /mnt/sagemaker-nvme/nlu_code
python3 src/data_preprocessing/data_preparation_pipeline.py
```

**Expected Output**:
```
INFO:__main__:Loaded 29 valid intents          ← Should show 29 now (not 9)
INFO:__main__:Loaded 24 valid entities         ← Should show 24 now
INFO:__main__:Valid: 10079, Invalid: 3         ← Should be ~99.97% valid
INFO:__main__:Split: Train=7055, Val=1512, Test=1512
```

---

### Step 5: Train Models
```bash
# Set cache location first
export HF_HOME=/mnt/sagemaker-nvme/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/sagemaker-nvme/.cache/huggingface

# Run training
python3 src/training/train_all_models.py
```

---

## Alternative: Use Smaller Model (If Still Out of Space)

If you still run out of space, use a smaller model. Edit `src/training/train_intent.py`:

**Change From:**
```python
model_name = "microsoft/mdeberta-v3-base"  # 1.33GB
```

**Change To:**
```python
model_name = "distilbert-base-multilingual-cased"  # 542MB
```

Or:
```python
model_name = "google/muril-base-cased"  # 891MB (optimized for Indian languages)
```

---

## Permanent Fix: Add to ~/.bashrc

```bash
echo 'export HF_HOME=/mnt/sagemaker-nvme/.cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/mnt/sagemaker-nvme/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc
```

---

## Verify Everything is Working

### 1. Check Validation Output
```bash
python3 src/data_preprocessing/data_preparation_pipeline.py 2>&1 | grep -E "Loaded|Valid:"
```

Should show:
```
INFO:__main__:Loaded 29 valid intents
INFO:__main__:Loaded 24 valid entities
INFO:__main__:Valid: 10079, Invalid: 3
```

### 2. Check Disk Space
```bash
df -h /mnt/sagemaker-nvme
```

Should have > 10GB free.

### 3. Check Intent Distribution
```bash
python3 src/data_preprocessing/data_preparation_pipeline.py 2>&1 | grep -A 30 "Train distribution"
```

Should show ALL your intents, not just `booking_status`.

---

## Expected Training Time

- **Data Preprocessing**: 1-2 minutes
- **Intent Classifier Training**: 30-45 minutes (10 epochs)
- **Entity Extractor Training**: 20-30 minutes
- **Total**: ~1 hour

---

## After Training Success

Your models will be saved to:
```
models/intent_classifier/
  ├── model.safetensors
  ├── config.json
  ├── tokenizer_config.json
  └── ...

models/entity_extractor/
  ├── model files...
  └── ...
```

---

## Troubleshooting

### Still showing 9,113 invalid?
- Make sure you pulled latest code: `git log --oneline -1`
- Should show: `5598d95d Fix validation: Load intents/entities dynamically`

### Still out of disk space?
1. Check which partition is full: `df -h`
2. Clear cache: `rm -rf /home/sagemaker-user/.cache/*`
3. Use smaller model (see alternative above)

### Import error: yaml
```bash
pip install pyyaml
```

---

## Next Steps After Training

1. ✅ **Test NLU Pipeline**
   ```bash
   python3 src/nlu/test_nlu_pipeline.py
   ```

2. ✅ **Start Building Backend APIs**
   - Use `INTENT_API_MAPPING.md` for endpoint specifications
   - Implement all 18 REST endpoints
   - Use Flask/FastAPI framework

3. ✅ **Test Integration**
   - Deploy API server
   - Connect NLU to backend
   - Test end-to-end flow

---

## Summary of Changes

| File | Change | Impact |
|------|--------|--------|
| `src/data_preprocessing/data_preparation_pipeline.py` | Load intents/entities from YAML config | Fixes 90% validation failure |
| `INTENT_API_MAPPING.md` | Complete intent-to-API mapping | Backend team can start implementation |
| `diagnose_and_fix_data.py` | Diagnostic script | Helps debug future issues |
| `data/raw/training_data_cleaned.json` | Cleaned data (99.97% valid) | Backup if needed |

---

**Questions?** Check logs or run diagnostic script:
```bash
python3 diagnose_and_fix_data.py
```
