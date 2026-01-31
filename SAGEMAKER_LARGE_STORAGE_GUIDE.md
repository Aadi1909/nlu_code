# SageMaker Training Guide - Large Storage Setup

## Overview
This guide shows how to train your Intent Classification model on AWS SageMaker with large storage support.

## Storage Options on SageMaker

### 1. **EBS Volume (Default)**
- Specified via `volume_size` parameter
- Range: 1GB - 16TB
- Persistent storage
- **Recommendation**: 100-500GB for your use case

### 2. **NVMe SSD (Instance Storage)**
- Available on specific instance types (ml.p3, ml.p4d, ml.g4dn)
- Much faster than EBS
- Automatically detected by the training script
- **Location**: `/mnt/sagemaker-nvme/`
- **Your scripts already support this!**

## Quick Start

### Step 1: Install SageMaker SDK (Local Machine)

```bash
pip install sagemaker boto3 awscli
aws configure  # Enter your AWS credentials
```

### Step 2: Prepare Your Data

```bash
# Your data is already processed!
cd /Users/divyansh.masand/Desktop/model_proj
python3 process_training_data.py  # Already done
```

### Step 3: Run SageMaker Training

```bash
# Option A: Use the provided script
python3 run_sagemaker_training.py

# Option B: Use AWS Console (see below)
```

## Instance Types for Large Storage

### Recommended Instances:

| Instance Type | GPUs | GPU RAM | Storage | Cost/hr | Use Case |
|--------------|------|---------|---------|---------|----------|
| **ml.g4dn.xlarge** | 1x T4 | 16GB | 125GB NVMe | $0.74 | Small-Medium datasets |
| **ml.g4dn.2xlarge** | 1x T4 | 32GB | 225GB NVMe | $0.94 | Medium datasets |
| **ml.p3.2xlarge** | 1x V100 | 16GB | No NVMe | $3.06 | Fast training |
| **ml.p3.8xlarge** | 4x V100 | 64GB | No NVMe | $12.24 | Multi-GPU |
| **ml.p4d.24xlarge** | 8x A100 | 320GB | 8TB NVMe | $32.77 | **Large datasets (16 lakh+)** |

**For 16 lakh rows**: Use **ml.p4d.24xlarge** or **ml.p3.8xlarge**

## Method 1: Using Python Script

Edit `run_sagemaker_training.py`:

```python
CONFIG = {
    'instance_type': 'ml.p4d.24xlarge',  # For large datasets
    'instance_count': 1,
    'volume_size': 500,  # 500GB EBS volume
    # ... rest of config
}
```

Then run:
```bash
python3 run_sagemaker_training.py
```

## Method 2: Using AWS SageMaker Console

### 1. Upload Data to S3

```bash
# Install AWS CLI
pip install awscli
aws configure

# Create S3 bucket (if needed)
aws s3 mb s3://your-bucket-name-intent-classifier

# Upload processed data
aws s3 sync data/processed/ s3://your-bucket-name-intent-classifier/data/
```

### 2. Upload Training Code

```bash
# Zip your source code
cd src
tar -czf ../training.tar.gz train_intent_classifier.py

# Upload to S3
aws s3 cp ../training.tar.gz s3://your-bucket-name-intent-classifier/code/
```

### 3. Create Training Job in Console

1. Go to **AWS Console** > **SageMaker** > **Training jobs** > **Create training job**

2. **Job Settings**:
   - Job name: `intent-classifier-training-{timestamp}`
   - IAM role: Select your SageMaker execution role

3. **Algorithm Settings**:
   - Algorithm source: **HuggingFace**
   - Container: Select HuggingFace PyTorch container
   
4. **Resource Configuration**:
   - Instance type: `ml.p4d.24xlarge` (or your choice)
   - Instance count: `1`
   - **Volume size: 500 GB** ⬅️ **Important for large storage**

5. **Hyperparameters**:
   ```
   epochs=15
   batch_size=32
   learning_rate=3e-5
   model_name=google/muril-base-cased
   ```

6. **Input Data**:
   - Channel name: `training`
   - S3 location: `s3://your-bucket-name-intent-classifier/data/`

7. **Output Data**:
   - S3 location: `s3://your-bucket-name-intent-classifier/output/`

8. **Click** "Create training job"

## Method 3: Using AWS CLI

Create a file `sagemaker-config.json`:

```json
{
  "TrainingJobName": "intent-classifier-20260131",
  "RoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerRole",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13-transformers4.26-gpu-py39",
    "TrainingInputMode": "File"
  },
  "ResourceConfig": {
    "InstanceType": "ml.p4d.24xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 500
  },
  "InputDataConfig": [
    {
      "ChannelName": "training",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://your-bucket/data/",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "s3://your-bucket/output/"
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 86400
  }
}
```

Run:
```bash
aws sagemaker create-training-job --cli-input-json file://sagemaker-config.json
```

## NVMe Storage Detection

Your training scripts **automatically detect and use NVMe storage**:

```python
# This code is already in your training scripts!
if os.path.exists("/mnt/sagemaker-nvme"):
    output_dir = "/mnt/sagemaker-nvme/intent_models"
    logger.info(f"Using NVMe storage: {output_dir}")
else:
    output_dir = "models/intent_classifier"
```

## Monitoring Training

### View Logs:
```bash
# Install SageMaker SDK
pip install sagemaker

# In Python:
import sagemaker
sess = sagemaker.Session()
sess.logs_for_job('your-job-name', wait=True)
```

### AWS Console:
1. Go to **SageMaker** > **Training jobs**
2. Click on your job name
3. View **CloudWatch logs**

## Cost Optimization Tips

### 1. Use Spot Instances (Save 70%)
```python
CONFIG = {
    'use_spot_instances': True,
    'max_wait': 90000,  # Max wait time
}
```

### 2. Stop Training Early
- Use early stopping (already configured)
- Monitor training progress

### 3. Choose Right Instance
- Start with `ml.g4dn.xlarge` for testing
- Scale up to `ml.p4d.24xlarge` for full dataset

## After Training

### Download Model:

```bash
# List training jobs
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending

# Download model artifacts
aws s3 cp s3://your-bucket/output/job-name/output/model.tar.gz .

# Extract
tar -xzf model.tar.gz
```

## Troubleshooting

### Out of Memory Error:
- Increase instance type
- Reduce batch size in hyperparameters
- Enable gradient checkpointing (already enabled)

### Storage Full:
- Increase `volume_size` parameter
- Use instance with NVMe storage
- Clear cache in training script

### Slow Training:
- Use GPU instance (ml.g4dn, ml.p3, ml.p4d)
- Enable mixed precision (already enabled)
- Increase batch size with gradient accumulation

## Files Ready for SageMaker:

✅ `src/train_intent_classifier.py` - Standard training (7K samples)
✅ `src/train_large_dataset.py` - Optimized for large datasets (100K+)
✅ `data/processed/` - Processed data ready to upload
✅ `run_sagemaker_training.py` - SageMaker launcher script

## Next Steps:

1. **Test locally first** (if you have GPU):
   ```bash
   cd src
   python3 train_intent_classifier.py
   ```

2. **Upload to S3**:
   ```bash
   aws s3 sync data/processed/ s3://your-bucket/data/
   ```

3. **Run on SageMaker**:
   ```bash
   python3 run_sagemaker_training.py
   ```

---

**Your training scripts are fully optimized and ready for SageMaker with large storage support!** 🚀
