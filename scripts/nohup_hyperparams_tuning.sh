#!/bin/bash

nohup python scripts/hyperparams_tuning.py \
    --mRNA_max_len 30 \
    --miRNA_max_len 24 \
    --device cuda:2 \
    --epochs 100 \
    --batch_size 64 \
    --base_model_name HyenaDNA \
    --model_name HyenaDNA_miRNA \
    --dataset_name TargetScan \
    --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train.csv \
    --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation.csv \
    --use_head \
    > optuna_logs/HyenaDNA_miRNAonly_TargetScan_short_w_linker_revmiRNA.log 2>&1 &