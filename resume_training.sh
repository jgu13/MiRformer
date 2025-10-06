#!/bin/bash

# Script to resume training from checkpoint with specific learning rate
# Usage: ./resume_training.sh <checkpoint_path> <resume_epoch>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <checkpoint_path> <resume_epoch>"
    echo "Example: $0 checkpoints/TargetScan/TwoTowerTransformer/Longformer/520/Pretrain_DDP/best_accuracy_0.1234_epoch3.pth 4"
    exit 1
fi

CHECKPOINT_PATH=$1
RESUME_EPOCH=$2

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
echo "Starting from epoch: $RESUME_EPOCH"
echo "Using learning rate: 0.0000968"

# Run the training script with checkpoint and resume parameters
CUDA_VISIBLE_DEVICES=1,2,3 nohup \
    torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    scripts/Pretrain.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --resume_epoch "$RESUME_EPOCH" \
    --lr 0.0000968 \
    > train_resume_epoch${RESUME_EPOCH}.out 2>&1 &

echo "Training resumed in background. Check train_resume_epoch${RESUME_EPOCH}.out for logs."
echo "Process ID: $!"


