#!/bin/bash

# Script to run multi-GPU training with Pretrain.py
# This script demonstrates how to launch distributed training

# Run the training script
CUDA_VISIBLE_DEVICES=1,2,3 nohup \
    torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    scripts/Pretrain.py \
    > train_Merged_500_randomized_start_DDP.out 2>&1 &
