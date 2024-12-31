#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# nohup python \
#     scripts/main_evaluate_hyenaDNA.py \
#     --mRNA_max_len 1000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/1000/checkpoint_epoch_final.pth \
#     > evaluate_logs/evaluate_Hyena_1000.log 2>&1 &

# nohup python \
#     scripts/main_evaluate_hyenaDNA.py \
#     --mRNA_max_len 2000 \
#     --device cuda:2 \
#     --batch_size 32 \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/2000/checkpoint_epoch_final.pth \
#     > evaluate_logs/evaluate_Hyena_2000.log 2>&1 &

# nohup python \
#     scripts/main_evaluate_hyenaDNA.py \
#     --mRNA_max_len 3000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/3000/checkpoint_epoch_final.pth \
#     > evaluate_logs/evaluate_Hyena_3000.log 2>&1 &

# nohup python \
#     scripts/main_evaluate_hyenaDNA.py \
#     --mRNA_max_len 4000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/4000/checkpoint_epoch_final.pth \
#     > evaluate_logs/evaluate_Hyena_4000.log 2>&1 &

# nohup python \
#     scripts/main_evaluate_hyenaDNA.py \
#     --mRNA_max_len 5000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/HyenaDNA/5000/checkpoint_epoch_final.pth \
#     > evaluate_logs/evaluate_Hyena_5000.log 2>&1 &

## evaluate miraw
# nohup python \
#     scripts/main_evaluate_hyenaDNA.py \
#     --mRNA_max_len 40 \
#     --device cuda:3 \
#     --batch_size 32 \
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/miRAW/HyenaDNA/checkpoint_epoch_final.pth \
#     > evaluate_logs/evaluate_Hyena_miraw.log 2>&1 &