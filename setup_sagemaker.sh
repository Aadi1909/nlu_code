#!/bin/bash
# Quick SageMaker Setup Script for Large Storage Training

echo "=========================================="
echo "SageMaker Training Setup - Large Storage"
echo "=========================================="
echo ""

# Configuration
BUCKET_NAME="intent-classifier-$(date +%Y%m%d)"
INSTANCE_TYPE="ml.g4dn.xlarge"  # Change to ml.p4d.24xlarge for large datasets
VOLUME_SIZE=500  # GB
REGION="us-east-1"

echo "📋 Configuration:"
echo "   S3 Bucket: $BUCKET_NAME"
echo "   Instance: $INSTANCE_TYPE"
echo "   Storage: ${VOLUME_SIZE}GB"
echo "   Region: $REGION"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found!"
    echo "   Install: pip install awscli"
    exit 1
fi

# Check AWS credentials
echo "1️⃣  Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured!"
    echo "   Run: aws configure"
    exit 1
fi
echo "✅ AWS credentials OK"

# Create S3 bucket
echo ""
echo "2️⃣  Setting up S3 bucket..."
if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    aws s3 mb "s3://$BUCKET_NAME" --region $REGION
    echo "✅ Created S3 bucket: $BUCKET_NAME"
else
    echo "✅ S3 bucket already exists: $BUCKET_NAME"
fi

# Upload processed data
echo ""
echo "3️⃣  Uploading training data to S3..."
if [ -d "data/processed" ]; then
    aws s3 sync data/processed/ "s3://$BUCKET_NAME/data/" --quiet
    echo "✅ Data uploaded to s3://$BUCKET_NAME/data/"
else
    echo "❌ data/processed/ not found!"
    echo "   Run: python3 process_training_data.py"
    exit 1
fi

# Package training code
echo ""
echo "4️⃣  Packaging training code..."
cd src
tar -czf ../training-code.tar.gz *.py
cd ..
aws s3 cp training-code.tar.gz "s3://$BUCKET_NAME/code/"
echo "✅ Code uploaded to s3://$BUCKET_NAME/code/"

# Display next steps
echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Your data is ready on S3:"
echo "   Data: s3://$BUCKET_NAME/data/"
echo "   Code: s3://$BUCKET_NAME/code/"
echo ""
echo "📝 Next steps:"
echo ""
echo "Option 1: Run via Python script"
echo "   python3 run_sagemaker_training.py"
echo ""
echo "Option 2: Run via AWS Console"
echo "   1. Go to: https://console.aws.amazon.com/sagemaker/"
echo "   2. Navigate to: Training > Training jobs > Create"
echo "   3. Configure:"
echo "      - Instance: $INSTANCE_TYPE"
echo "      - Volume: ${VOLUME_SIZE}GB"
echo "      - Data: s3://$BUCKET_NAME/data/"
echo "      - Output: s3://$BUCKET_NAME/output/"
echo ""
echo "Option 3: Run via AWS CLI"
echo "   See SAGEMAKER_LARGE_STORAGE_GUIDE.md"
echo ""
echo "💰 Cost estimate:"
case $INSTANCE_TYPE in
    "ml.g4dn.xlarge")
        echo "   ~$0.74/hour"
        ;;
    "ml.g4dn.2xlarge")
        echo "   ~$0.94/hour"
        ;;
    "ml.p3.2xlarge")
        echo "   ~$3.06/hour"
        ;;
    "ml.p4d.24xlarge")
        echo "   ~$32.77/hour"
        ;;
esac
echo ""
echo "🔍 Monitor training:"
echo "   AWS Console > SageMaker > Training jobs"
echo ""
