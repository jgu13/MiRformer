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

# # evaluate miraw
# nohup python \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name miRAW_noL_noMissing \
#     --test_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_test.csv \
#     --evaluate \
#     > evaluate_logs/evaluate_Hyena_miRAW_noL_noMissing.log 2>&1 &

# evaluate synthetic dataset
# nohup python \
#     scripts/main_finetune_Hyena_ddp.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name selected_perfect_seed_match\
#     --test_dataset_path $data_dir/selected_perfect_seed_match_test.csv \
#     --evaluate \
#     > evaluate_logs/evaluate_Hyena_selected_perfect_seed_match.log 2>&1 &

# evaluate selected seed match dataset
nohup python \
    scripts/main_finetune_Hyena_ddp.py \
    --mRNA_max_len 40 \
    --miRNA_max_len 26 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_name selected_perfect_seed_match_w_linker \
    --test_dataset_path $data_dir/ISM_data_mutant_miRNA_nonseed.csv \
    --evaluate \
    > evaluate_logs/evaluate_Hyena_ISM_mutant_miRNA_nonseed.log 2>&1 &