#!/bin/bash

data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# nohup python scripts/baseline_CNN.py --mRNA_max_len 1000 --device cuda:9 --num_epochs 50 --batch_size 128 > output_logs/output_CNN_1000_50.log 2>&1 &
# nohup python scripts/baseline_CNN.py --mRNA_max_len 2000 --device cuda:2 --num_epochs 50 --batch_size 256 > output_logs/output_CNN_2000_50.log 2>&1 &
# mutli-gpu training
# CUDA_VISIBLE_DEVICES=3,4 nohup \
#     torchrun \
#     --nproc_per_node=2 \
#     --master_port=29505 \
#     scripts/baseline_CNN.py \
#     --mRNA_max_len 3000 \
#     --num_epochs 50 \
#     --batch_size 128 \
#     --ddp \
#     > output_logs/output_CNN_3000_50.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6,7 nohup \
#     torchrun \
#     --nproc_per_node=3 \
#     --master_port=29506 \
#     scripts/baseline_CNN.py \
#     --mRNA_max_len 4000 \
#     --num_epochs 50 \
#     --batch_size 128 \
#     --ddp \
#     > output_logs/output_CNN_4000_50.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1,2 nohup \
#     torchrun \
#     --nproc_per_node=3 \
#     --master_port=29507 \
#     scripts/baseline_CNN.py \
#     --mRNA_max_len 5000 \
#     --num_epochs 50 \
#     --batch_size 32 \
#     --ddp \
#     > output_logs/output_CNN_5000_50.log 2>&1 &

# nohup python scripts/baseline_CNN.py --mRNA_max_len 4000 --device cuda:6 --batch_size 256 --num_epochs 50 > output_logs/output_CNN_4000_50.log 2>&1 &
# nohup python scripts/baseline_CNN.py --mRNA_max_len 5000 --device cuda:2 --batch_size 256 --num_epochs 50 > output_logs/output_CNN_5000_50.log 2>&1 &

# train on miRAW dataset
nohup python scripts/baseline_CNN.py \
    --mRNA_max_len 40 \
    --device cuda:7 \
    --num_epochs 50 \
    --batch_size 64 \
    --dataset_name miRAW \
    --dataset_path $data_dir/miRAW_dataset.csv\
    > output_logs/output_CNN_miRAW_50.log 2>&1 & # requires parameter changes before running