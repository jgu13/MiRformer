#!/bin/bash

# Script to run multi-GPU training with Pretrain.py
# This script demonstrates how to launch distributed training

# Run the training script
CUDA_VISIBLE_DEVICES=2,3 nohup \
    torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    Pretrain.py \
    > train_Positive_Primates_500_randomized_start_DDP.out 2>&1 &
