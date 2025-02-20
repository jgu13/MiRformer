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

# evaluate MLP on miRAW
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
#     --mRNA_max_length 40 \
#     --miRNA_max_length 26 \
#     --device cuda:3 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTowerMLP \
#     --dataset_name miRAW_noL_noMissing \
#     --test_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_test.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
#     --accumulation_step 2 \
#     --evaluate \
#     > evaluate_logs/evaluate_TwoTowerMLP_miRaw_noL_noMissing_revmiRNA.log 2>&1 &

# evaluate MLP trained on selected perfect seed match
# nohup python scripts/main.py\
#     --mRNA_max_length 40 \
#     --miRNA_max_length 26 \
#     --device cuda:3 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTowerMLP \
#     --dataset_name selected_perfect_seed_match \
#     --test_dataset_path $data_dir/selected_perfect_seed_match_test.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
#     --accumulation_step 2 \
#     --evaluate \
#     > evaluate_logs/evaluate_TwoTowerMLP_selcted_perfect_seed_match.log 2>&1 &

# evaluate MLP on ISM data
nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
    --mRNA_max_length 40 \
    --miRNA_max_length 26 \
    --device cuda:3 \
    --epochs 100 \
    --batch_size 64 \
    --base_model_name TwoTowerMLP \
    --model_name TwoTowerMLP \
    --dataset_name miRAW_noL_noMissing \
    --test_dataset_path $data_dir/ISM_data_mutant_miRNA_seed.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
    --accumulation_step 2 \
    --evaluate \
    > evaluate_logs/evaluate_TwoTowerMLP_ISM_miRNA_seed.log 2>&1 &
