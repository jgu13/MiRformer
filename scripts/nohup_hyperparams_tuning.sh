#!/bin/bash
# nohup python scripts/hyperparams_tuning.py \
#     --mRNA_max_len 500 \
#     --miRNA_max_len 24 \
#     --device cuda:1 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_miRNA_500_2 \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_500.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_500.csv \
#     --use_head \
#     > optuna_logs/HyenaDNA_miRNAonly_TargetScan_500_2.log 2>&1 &


# nohup python scripts/hyperparams_tuning.py \
#     --mRNA_max_len 924 \
#     --miRNA_max_len 24 \
#     --device cuda:1 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_miRNA_924 \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_924.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_924.csv \
#     --use_head \
#     > optuna_logs/HyenaDNA_miRNAonly_TargetScan_924_w_linker_revmiRNA.log 2>&1 &

nohup python scripts/hyperparams_tuning.py \
    --mRNA_max_len 10000 \
    --miRNA_max_len 24 \
    --device cuda:1 \
    --epochs 100 \
    --batch_size 4 \
    --base_model_name HyenaDNA \
    --model_name HyenaDNA_miRNA_10k \
    --dataset_name TargetScan \
    --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_10k.csv \
    --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_10k.csv \
    --use_head \
    > optuna_logs/HyenaDNA_miRNAonly_TargetScan_10k_w_linker_revmiRNA.log 2>&1 &

# nohup python scripts/hyperparams_tuning.py \
#     --mRNA_max_len 22552 \
#     --miRNA_max_len 24 \
#     --device cuda:1 \
#     --epochs 100 \
#     --batch_size 8 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_miRNA_full_length \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_full_length_train.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_full_length_validation.csv \
#     --use_head \
#     > optuna_logs/HyenaDNA_miRNAonly_TargetScan_full_length_w_linker_revmiRNA.log 2>&1 &