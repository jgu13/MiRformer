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

# nohup python scripts/main_evaluate_MLP.py \
#     --mRNA_max_len 3000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name mirLM \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     > evaluate_logs/evaluate_MLP_3000.log 2>&1 &

# nohup python scripts/main_evaluate_MLP.py \
#     --mRNA_max_len 4000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name mirLM \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     > evaluate_logs/evaluate_MLP_4000.log 2>&1 &

# nohup python scripts/main_evaluate_MLP.py \
#     --mRNA_max_len 5000 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name mirLM \
#     --dataset_path $data_dir/DeepMirTar-par-clip-selected.csv \
#     > evaluate_logs/evaluate_MLP_5000.log 2>&1 &

## evaluate MLP trained on miRAW
# nohup python scripts/main_MLP.py\
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name miRAW_noL_noMissing\
#     --test_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_test.csv \
#     --evaluate \
#     > evaluate_logs/evaluate_MLP_miRAW_noL_noMissing.log 2>&1 &

# evaluate MLP trained on synthetic_dataset
# nohup python scripts/main_MLP.py\
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26 \
#     --device cuda:2 \
#     --batch_size 64 \
#     --dataset_name selected_perfect_seed_match\
#     --test_dataset_path $data_dir/selected_perfect_seed_match_test.csv \
#     --evaluate \
#     > evaluate_logs/evaluate_MLP_selected_perfect_seed_match.log 2>&1 &

# evaluate MLP trained on ISM data
nohup python scripts/main_MLP.py\
    --mRNA_max_len 40 \
    --miRNA_max_len 26 \
    --device cuda:2 \
    --batch_size 64 \
    --dataset_name selected_perfect_seed_match\
    --test_dataset_path $data_dir/ISM_data_mutant_miRNA_nonseed.csv \
    --evaluate \
    > evaluate_logs/evaluate_MLP_ISM_data_mutant_miRNA_nonseed.log 2>&1 &
