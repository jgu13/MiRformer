#!/bin/bash
# nohup python \
#     scripts/main_evaluate_CNN.py \
#     --mRNA_max_len 1000 \
#     --device cuda:2 \
#     --batch_size 128 \
#     --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-selected.csv\
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/1000/checkpoint_final.pth \
#     > evaluate_logs/evaluate_CNN_1000.log 2>&1 &

# nohup python \
#     scripts/main_evaluate_CNN.py \
#     --mRNA_max_len 2000 \
#     --device cuda:2 \
#     --batch_size 32 \
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/2000/checkpoint_final.pth \
#     > evaluate_logs/evaluate_CNN_2000.log 2>&1 &

# nohup python \
#     scripts/main_evaluate_CNN.py \
#     --mRNA_max_len 3000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-selected.csv\
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/3000/checkpoint_final_corrected.pth \
#     > evaluate_logs/evaluate_CNN_3000.log 2>&1 &

nohup python \
    scripts/main_evaluate_CNN.py \
    --mRNA_max_len 4000 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-selected.csv\
    --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/4000/checkpoint_final.pth \
    > evaluate_logs/evaluate_CNN_4000.log 2>&1 &

nohup python \
    scripts/main_evaluate_CNN.py \
    --mRNA_max_len 5000 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-selected.csv\
    --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/5000/checkpoint_final.pth \
    > evaluate_logs/evaluate_CNN_5000.log 2>&1 &

## evalute miraw-trained CNN
# nohup python \
#     scripts/main_evaluate_CNN.py \
#     --mRNA_max_len 40 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-miraw-like.csv\
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/miRAW/CNN/checkpoint_final.pth \
#     > evaluate_logs/evaluate_CNN_miraw.log 2>&1 &