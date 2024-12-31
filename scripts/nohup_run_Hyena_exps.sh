#!/bin/bash

data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# CUDA_VISIBLE_DEVICES=0,1 nohup \
#     torchrun --nproc_per_node=2 \
#     --master_port=29500 \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 1000 \
#     --num_epochs 30 \
#     --ddp \
#     --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/1000/checkpoint_epoch_20.pth \
#     > output_logs/output_hyena_1000_continued_30.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup \
#     torchrun --nproc_per_node=2 \
#     --master_port=29501 \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 2000 \
#     --num_epochs 40 \
#     --batch_size 16 \
#     --ddp \
#     --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/2000/checkpoint_epoch_10.pth \
#     > output_logs/output_hyena_2000_continued_40.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4,5 nohup torchrun \
#     --nproc_per_node=2 \
#     --master_port=29502 \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 3000 \
#     --num_epochs 40 \
#     --batch_size 16 \
#     --ddp \
#     --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/3000/checkpoint_epoch_10.pth \
#     > output_logs/output_hyena_3000_continued_40.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6,7,8 nohup \
#     torchrun --nproc_per_node=3 \
#     --master_port=29503 \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 4000 \
#     --num_epochs 40 \
#     --batch_size 16 \
#     --ddp \
#     --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/4000/checkpoint_epoch_10.pth \
#     > output_logs/output_hyena_4000_continued_40.log 2>&1 &

# # on mcb server
# CUDA_VISIBLE_DEVICES=0,1,2 nohup torchrun \
#     --nproc_per_node=3 \
#     --master_port=29504 \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 5000 \
#     --num_epochs 40 \
#     --batch_size 16 \
#     --ddp \
#     --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/5000/checkpoint_epoch_10.pth \
#     > output_logs/output_hyena_5000_continued_40.log 2>&1 &

# train on miRAW data
# nohup python \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 40\
#     --device cuda:1\
#     --num_epochs 50 \
#     --batch_size 64\
#     --dataset_name miRAW \
#     --dataset_path $data_dir/miRAW_dataset.csv \
#     > output_logs/output_hyena_miraw_50.log 2>&1 &