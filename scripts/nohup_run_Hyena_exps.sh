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

# train on miRAW data
# nohup python \
#     scripts/main.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26\
#     --device cuda:2 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA \
#     --dataset_name miRAW_noL_noMissing \
#     --train_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_train.csv \
#     --val_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_validation.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig.json \
#     --use_head \
#     --accumulation_step 8 \
#     > output_logs/output_HyenaDNA_miRaw_noL_noMissing_revmiRNA.log 2>&1 &

# continue train on selected perfect seed match
# nohup python \
#     scripts/main.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26\
#     --device cuda:2 \
#     --epochs 50 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA \
#     --dataset_name selected_perfect_seed_match \
#     --train_dataset_path $data_dir/selected_perfect_seed_match_train.csv \
#     --val_dataset_path $data_dir/selected_perfect_seed_match_validation.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig.json \
#     --use_head \
#     --accumulation_step 8 \
#     --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/miRAW_noL_noMissing/HyenaDNA/40/checkpoint_epoch_final.pth\
#     > output_logs/output_HyenaDNA_selected_perfect_seed_match.log 2>&1 &

# Train on TargetScan dataset
nohup python \
    scripts/main.py \
    --mRNA_max_len 30 \
    --miRNA_max_len 26\
    --device cuda:2 \
    --epochs 100 \
    --batch_size 64 \
    --base_model_name HyenaDNA \
    --model_name HyenaDNA_miRNA \
    --dataset_name TargetScan \
    --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train.csv \
    --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig.json \
    --use_head \
    --accumulation_step 8 \
    > output_logs/output_HyenaDNA_TargetScan_short.log 2>&1 &