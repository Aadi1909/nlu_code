#!/bin/bash
# Quick fix for SageMaker disk space issue

echo "=== SageMaker Disk Space Fix ==="
echo ""
echo "Step 1: Check current disk usage"
df -h | grep -E "Filesystem|nvme|home"
echo ""

echo "Step 2: Move to NVMe disk and clone fresh"
echo "----------------------------------------"

# Create workspace on NVMe
mkdir -p /mnt/sagemaker-nvme/workspace
cd /mnt/sagemaker-nvme/workspace

# Clone the repo fresh (if not already there)
if [ ! -d "nlu_code" ]; then
    echo "Cloning repository..."
    git clone https://github.com/Aadi1909/nlu_code.git
else
    echo "Repository already exists, pulling latest changes..."
    cd nlu_code
    git pull origin main
    cd ..
fi

echo ""
echo "✅ Done! Your workspace is now at:"
echo "/mnt/sagemaker-nvme/workspace/nlu_code"
echo ""
echo "To use it, run:"
echo "cd /mnt/sagemaker-nvme/workspace/nlu_code/src"
echo "python train_intent_classifier.py"
