#!/usr/bin/env python3
"""
SageMaker Training Script for Intent Classification

This script sets up and runs training on AWS SageMaker with:
- Large storage (NVMe support)
- GPU instances (ml.g4dn, ml.p3, ml.p4d)
- Proper data handling
"""

import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
import os
from datetime import datetime

# Configuration
CONFIG = {
    # Instance Configuration
    'instance_type': 'ml.g4dn.xlarge',  # Options: ml.g4dn.xlarge, ml.p3.2xlarge, ml.p4d.24xlarge
    'instance_count': 1,
    'volume_size': 100,  # GB - EBS volume size
    
    # Training Configuration
    'max_run': 86400,  # 24 hours in seconds
    'use_spot_instances': False,  # Set to True for cost savings
    'checkpoint_s3_uri': None,  # Will be set automatically
    
    # Hyperparameters
    'hyperparameters': {
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 3e-5,
        'model_name': 'google/muril-base-cased',
        'max_length': 128,
    }
}

def setup_sagemaker_session():
    """Initialize SageMaker session and get role"""
    try:
        role = get_execution_role()
    except:
        # If running locally, specify your SageMaker execution role ARN
        role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE"
        print(f"⚠️  Using manually specified role: {role}")
        print("    Update this with your actual SageMaker execution role ARN")
    
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    
    return sess, role, bucket


def upload_data_to_s3(sess, bucket, local_data_dir='data/processed'):
    """Upload processed data to S3"""
    print(f"\n📤 Uploading data to S3...")
    
    # Upload data directory to S3
    s3_data_path = f"s3://{bucket}/intent-classification/data"
    s3_uri = sess.upload_data(
        path=local_data_dir,
        bucket=bucket,
        key_prefix='intent-classification/data'
    )
    
    print(f"✅ Data uploaded to: {s3_uri}")
    return s3_uri


def create_training_script():
    """Ensure training script is ready"""
    # The training script will be in src/train_intent_classifier.py
    # SageMaker will use this as the entry point
    return 'src/train_intent_classifier.py'


def run_training(sess, role, bucket, s3_data_uri, instance_type='ml.g4dn.xlarge'):
    """Run SageMaker training job"""
    
    # Create output path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path = f"s3://{bucket}/intent-classification/output/{timestamp}"
    checkpoint_path = f"s3://{bucket}/intent-classification/checkpoints/{timestamp}"
    
    print(f"\n🚀 Starting SageMaker Training Job")
    print(f"   Instance: {instance_type}")
    print(f"   Output: {output_path}")
    print(f"   Checkpoints: {checkpoint_path}")
    
    # Create HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point='train_intent_classifier.py',
        source_dir='src',
        instance_type=instance_type,
        instance_count=CONFIG['instance_count'],
        role=role,
        transformers_version='4.26',
        pytorch_version='1.13',
        py_version='py39',
        hyperparameters=CONFIG['hyperparameters'],
        
        # Storage configuration
        volume_size=CONFIG['volume_size'],  # EBS volume in GB
        
        # Output configuration
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_path,
        
        # Time limits
        max_run=CONFIG['max_run'],
        
        # Spot instances (optional - for cost savings)
        use_spot_instances=CONFIG['use_spot_instances'],
        max_wait=CONFIG['max_run'] + 3600 if CONFIG['use_spot_instances'] else None,
        
        # Metric definitions for tracking
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'loss: ([0-9\\.]+)'},
            {'Name': 'eval:loss', 'Regex': 'eval_loss: ([0-9\\.]+)'},
            {'Name': 'eval:accuracy', 'Regex': 'eval_accuracy: ([0-9\\.]+)'},
            {'Name': 'eval:f1', 'Regex': 'eval_f1: ([0-9\\.]+)'},
        ],
        
        # Environment variables
        environment={
            'TRANSFORMERS_CACHE': '/tmp/transformers_cache',
        }
    )
    
    # Start training
    print("\n⏳ Starting training job...")
    huggingface_estimator.fit({'training': s3_data_uri}, wait=True)
    
    print(f"\n✅ Training complete!")
    print(f"   Model artifacts: {output_path}")
    print(f"   Job name: {huggingface_estimator.latest_training_job.name}")
    
    return huggingface_estimator


def download_model(estimator, local_dir='models/sagemaker_trained'):
    """Download trained model from S3"""
    print(f"\n📥 Downloading trained model...")
    
    import os
    os.makedirs(local_dir, exist_ok=True)
    
    # Download model artifacts
    model_data = estimator.model_data
    print(f"   Model location: {model_data}")
    print(f"   Downloading to: {local_dir}")
    
    # You can download manually or use:
    # aws s3 cp <model_data> <local_dir> --recursive
    
    print(f"\n💡 To download model, run:")
    print(f"   aws s3 cp {model_data} {local_dir}/ --recursive")


def main():
    """Main execution function"""
    
    print("="*70)
    print("AWS SAGEMAKER TRAINING FOR INTENT CLASSIFICATION")
    print("="*70)
    
    # 1. Setup SageMaker
    print("\n1️⃣  Setting up SageMaker session...")
    sess, role, bucket = setup_sagemaker_session()
    print(f"   Role: {role}")
    print(f"   Bucket: {bucket}")
    
    # 2. Upload data to S3
    print("\n2️⃣  Uploading data to S3...")
    s3_data_uri = upload_data_to_s3(sess, bucket)
    
    # 3. Select instance type
    print("\n3️⃣  Instance Selection:")
    print("   Available options:")
    print("   - ml.g4dn.xlarge   : 1 GPU (T4), 16GB GPU RAM, $0.736/hr  [Recommended]")
    print("   - ml.g4dn.2xlarge  : 1 GPU (T4), 32GB GPU RAM, $0.94/hr")
    print("   - ml.p3.2xlarge    : 1 GPU (V100), 16GB GPU RAM, $3.06/hr")
    print("   - ml.p3.8xlarge    : 4 GPU (V100), 64GB GPU RAM, $12.24/hr")
    print("   - ml.p4d.24xlarge  : 8 GPU (A100), 320GB GPU RAM, $32.77/hr [Large datasets]")
    
    instance_type = CONFIG['instance_type']
    print(f"\n   Using: {instance_type}")
    
    # 4. Run training
    print("\n4️⃣  Running training job...")
    estimator = run_training(sess, role, bucket, s3_data_uri, instance_type)
    
    # 5. Download model (optional)
    print("\n5️⃣  Model download information:")
    download_model(estimator)
    
    print("\n" + "="*70)
    print("✅ SAGEMAKER TRAINING SETUP COMPLETE!")
    print("="*70)
    print("\n💡 Tips:")
    print("   - Monitor training: AWS Console > SageMaker > Training Jobs")
    print("   - View logs: CloudWatch Logs")
    print(f"   - Use spot instances to save up to 70% on costs")
    print("   - Increase volume_size if you need more storage")
    
    return estimator


if __name__ == "__main__":
    # Check if running in SageMaker or locally
    if os.environ.get('SM_TRAINING_ENV'):
        print("Running inside SageMaker - use train_intent_classifier.py directly")
    else:
        print("Setting up SageMaker training job from local environment...")
        estimator = main()
