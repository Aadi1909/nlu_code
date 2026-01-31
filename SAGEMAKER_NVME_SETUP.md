# SageMaker NVMe Disk Setup Guide

## Problem
The home directory `/home/sagemaker-user` is only 5GB and fills up during model training, causing "No space left on device" errors.

## Solution
Use the large NVMe disk at `/mnt/sagemaker-nvme` (233GB available).

---

## Option 1: Move Entire Workspace (RECOMMENDED)

### Step 1: Move workspace to NVMe disk
```bash
# Create workspace directory on NVMe
mkdir -p /mnt/sagemaker-nvme/workspace

# Copy your code to NVMe
cp -r /home/sagemaker-user/workspace/nlu_code /mnt/sagemaker-nvme/workspace/

# Backup old location (optional)
mv /home/sagemaker-user/workspace/nlu_code /home/sagemaker-user/workspace/nlu_code.backup

# Create symbolic link for convenience (optional)
ln -s /mnt/sagemaker-nvme/workspace/nlu_code /home/sagemaker-user/workspace/nlu_code
```

### Step 2: Run training from NVMe
```bash
cd /mnt/sagemaker-nvme/workspace/nlu_code/src
python train_intent_classifier.py
```

**Benefits:**
- All files (code, data, models, checkpoints) are on the big disk
- No disk space issues
- Faster I/O operations

---

## Option 2: Keep Code in Place, Use Updated Script

The updated `train_intent_classifier.py` automatically detects SageMaker and uses NVMe disk:

```bash
# Just run training as normal
cd /home/sagemaker-user/workspace/nlu_code/src
python train_intent_classifier.py
```

**What happens:**
1. Script detects `/mnt/sagemaker-nvme` exists
2. Automatically uses `/mnt/sagemaker-nvme/intent_models` for all outputs
3. Saves checkpoints and final model to NVMe disk

---

## Verify Disk Space

Before training:
```bash
df -h
```

You should see:
```
/dev/nvme2n1    5.0G  XXX  XXX  XX% /home/sagemaker-user     ← Small disk
/dev/nvme1n1    233G   15G  219G   7% /mnt/sagemaker-nvme      ← Big disk!
```

---

## Quick Start Commands

```bash
# Option A: Move everything
mkdir -p /mnt/sagemaker-nvme/workspace
cp -r ~/workspace/nlu_code /mnt/sagemaker-nvme/workspace/
cd /mnt/sagemaker-nvme/workspace/nlu_code/src
python train_intent_classifier.py

# Option B: Use updated script (sync the updated train_intent_classifier.py first)
cd ~/workspace/nlu_code/src
python train_intent_classifier.py
```

---

## What Changed in train_intent_classifier.py

1. **Automatic NVMe Detection** (lines ~178-187):
   ```python
   if os.path.exists("/mnt/sagemaker-nvme"):
       output_dir = "/mnt/sagemaker-nvme/intent_models"
   else:
       output_dir = self.model_config.output_dir
   ```

2. **Fixed Parameters**:
   - `eval_strategy` instead of `evaluation_strategy` (transformers 4.30+)
   - All outputs go to NVMe when available

3. **Key Changes**:
   - TrainingArguments uses NVMe disk for checkpoints
   - Final model save uses same directory
   - No more split between checkpoint and final save locations

---

## Troubleshooting

### Still getting disk space errors?
```bash
# Check what's using space
du -sh /home/sagemaker-user/* | sort -h

# Clean up old checkpoints
rm -rf /home/sagemaker-user/workspace/nlu_code/models/intent_classifier/checkpoint-*

# Clean up cache
rm -rf ~/.cache/huggingface/hub/*
```

### Model saved but can't find it?
```bash
# Check NVMe disk
ls -lh /mnt/sagemaker-nvme/intent_models/

# Models are in:
# /mnt/sagemaker-nvme/intent_models/checkpoint-*  (intermediate)
# /mnt/sagemaker-nvme/intent_models/              (final model)
```

---

## After Training

Your trained model will be at:
- **On SageMaker**: `/mnt/sagemaker-nvme/intent_models/`
- Files: `model.safetensors`, `config.json`, `tokenizer.json`, `label_mapping.json`

To use the model:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/mnt/sagemaker-nvme/intent_models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```
