#!/bin/bash
# setup_bedrock.sh - Setup script for AWS Bedrock training

echo "=================================================="
echo "Battery Smart NLU - AWS Bedrock Setup"
echo "=================================================="

# 1. Update system
echo "1. Updating system packages..."
sudo apt-get update -y

# 2. Install Python dependencies
echo "2. Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify GPU availability
echo "3. Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 4. Verify data
echo "4. Verifying augmented training data..."
ls -lh data/processed/*_augmented.json

# 5. Ready to train
echo ""
echo "=================================================="
echo "✅ Setup complete! Ready to train."
echo "=================================================="
echo ""
echo "To start training:"
echo "  cd src"
echo "  python3 train_intent_classifier.py"
echo ""
echo "Expected training time: 3-5 minutes on GPU"
echo "Expected accuracy: 70%+"
echo "=================================================="
