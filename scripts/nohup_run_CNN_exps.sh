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
# nohup python scripts/baseline_CNN.py \
#     --mRNA_max_len 40 \
#     --device cuda:2 \
#     --num_epochs 10 \
#     --batch_size 64 \
#     --dataset_name miRAW \
#     --dataset_path $data_dir/miRAW_dataset.csv\
#     > output_logs/output_CNN_miRAW_50.log 2>&1 & # requires parameter changes before running

# train on completely random miRAW dataset
# nohup python scripts/baseline_CNN.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:1 \
#     --num_epochs 100 \
#     --batch_size 64 \
#     --dataset_name miRAW_noL_noMissing \
#     --train_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_train.csv \
#     --val_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_validation.csv \
#     > output_logs/output_CNN_miRaw_noL_noMissing.log 2>&1 & # requires parameter changes before running

# continue train on selceted perfect seed match 
# nohup python scripts/baseline_CNN.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:2 \
#     --num_epochs 50 \
#     --batch_size 64 \
#     --dataset_name selected_perfect_seed_match \
#     --train_dataset_path $data_dir/selected_perfect_seed_match_train.csv \
#     --val_dataset_path $data_dir/selected_perfect_seed_match_validation.csv \
#     --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/miRAW_noL_noMissing/CNN/40/checkpoint_epoch_final.pth \
#     > output_logs/output_CNN_selected_perfect_seed_match.log 2>&1 &

# continue train on TargetScan
# nohup python scripts/baseline_CNN.py \
#     --mRNA_max_len 994 \
#     --miRNA_max_len 24 \
#     --device cuda:1 \
#     --num_epochs 100 \
#     --batch_size 64 \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_1024_train.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_1024_validation.csv \
#     --accumulation_step 2\
#     > output_logs/output_CNN_TargetScan_w_linker_revmiRNA.log 2>&1 &