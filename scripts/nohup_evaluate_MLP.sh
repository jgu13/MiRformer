#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data
# nohup python scripts/main_evaluate_MLP.py \
#     --mRNA_max_len 1000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name mirLM \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     > evaluate_logs/evaluate_MLP_1000.log 2>&1 &

# nohup python scripts/main_evaluate_MLP.py \
#     --mRNA_max_len 2000 \
#     --device cuda:2 \
#     --batch_size 32 \
#     --dataset_name mirLM \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     > evaluate_logs/evaluate_MLP_2000.log 2>&1 &

nohup python scripts/main_evaluate_MLP.py \
    --mRNA_max_len 3000 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_name mirLM \
    --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
    > evaluate_logs/evaluate_MLP_3000.log 2>&1 &

nohup python scripts/main_evaluate_MLP.py \
    --mRNA_max_len 4000 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_name mirLM \
    --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
    > evaluate_logs/evaluate_MLP_4000.log 2>&1 &

nohup python scripts/main_evaluate_MLP.py \
    --mRNA_max_len 5000 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_name mirLM \
    --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
    > evaluate_logs/evaluate_MLP_5000.log 2>&1 &

## evaluate MLP trained on miRAW
# nohup python scripts/main_evaluate_MLP.py --mRNA_max_len 40 --device cuda:3 --batch_size 32 > evaluate_logs/evaluate_MLP_miraw.log 2>&1 &
