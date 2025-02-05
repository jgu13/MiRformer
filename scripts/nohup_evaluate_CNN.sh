#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data
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

# nohup python \
#     scripts/main_evaluate_CNN.py \
#     --mRNA_max_len 4000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-selected.csv\
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/4000/checkpoint_final.pth \
#     > evaluate_logs/evaluate_CNN_4000.log 2>&1 &

# nohup python \
#     scripts/main_evaluate_CNN.py \
#     --mRNA_max_len 5000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-selected.csv\
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/mirLM/CNN/5000/checkpoint_final.pth \
#     > evaluate_logs/evaluate_CNN_5000.log 2>&1 &

## evalute miraw-trained CNN
# nohup python \
#     scripts/main_evaluate_CNN.py \
#     --mRNA_max_len 40 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_path /home/mcb/users/jgu13/projects/mirLM/data/DeepMirTar-par-clip-miraw-like.csv\
#     --ckpt_path /home/mcb/users/jgu13/projects/mirLM/checkpoints/miRAW/CNN/checkpoint_final.pth \
#     > evaluate_logs/evaluate_CNN_miraw.log 2>&1 &

# # evalute miraw-random-trained CNN
# nohup python \
#     scripts/baseline_CNN.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name miRAW_noL_noMissing \
#     --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/data/data_miRaw_noL_noMissing_remained_seed1122_test.csv \
#     --evaluate \
#     > evaluate_logs/evaluate_CNN_miRAW_noL_noMissing.log 2>&1 &

# evalute synthetic dataset
# nohup python \
#     scripts/baseline_CNN.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name selected_perfect_seed_match\
#     --test_dataset_path $data_dir/selected_perfect_seed_match_test.csv \
#     --evaluate \
#     > evaluate_logs/evaluate_CNN_selected_perfect_seed_match.log 2>&1 &

nohup python \
    scripts/baseline_CNN.py \
    --mRNA_max_len 40 \
    --miRNA_max_len 26 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_name selected_perfect_seed_match\
    --test_dataset_path $data_dir/ISM_data_mutant_miRNA_nonseed.csv \
    --evaluate \
    > evaluate_logs/evaluate_CNN_ISM_data_mutant_miRNA_nonseed.log 2>&1 &