#!/bin/bash

# Script to move workspace to NVMe disk on SageMaker

echo "Moving workspace to NVMe disk..."

# Create directory on NVMe
mkdir -p /mnt/sagemaker-nvme/workspace

# Copy the entire nlu_code folder to NVMe
echo "Copying files..."
cp -r /home/sagemaker-user/workspace/nlu_code /mnt/sagemaker-nvme/workspace/

# Create a symbolic link from old location to new location (optional, for convenience)
echo "Creating symbolic link..."
mv /home/sagemaker-user/workspace/nlu_code /home/sagemaker-user/workspace/nlu_code.backup
ln -s /mnt/sagemaker-nvme/workspace/nlu_code /home/sagemaker-user/workspace/nlu_code

echo "Done! Workspace moved to /mnt/sagemaker-nvme/workspace/nlu_code"
echo "Old workspace backed up at /home/sagemaker-user/workspace/nlu_code.backup"
echo ""
echo "You can now run training from: /mnt/sagemaker-nvme/workspace/nlu_code/src"
