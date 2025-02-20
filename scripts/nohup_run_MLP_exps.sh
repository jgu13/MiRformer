#!/bin/bash

data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 1000 --device cuda:0 --num_epochs 50 > output_logs/output_MLP_1000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 2000 --device cuda:0 --num_epochs 50 > output_logs/output_MLP_2000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 3000 --device cuda:0 --num_epochs 50 > output_logs/output_MLP_3000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 4000 --device cuda:1 --num_epochs 50 > output_logs/output_MLP_4000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 5000 --device cuda:2 --num_epochs 50 > output_logs/output_MLP_5000_50_reverse.log 2>&1 &

# train on miRAW dataset
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
#     --mRNA_max_length 40 \
#     --miRNA_max_length 26 \
#     --device cuda:3 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTowerMLP \
#     --dataset_name miRAW_noL_noMissing \
#     --train_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_train.csv \
#     --val_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_validation.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
#     --accumulation_step 2 \
#     > output_logs/output_TwoTowerMLP_miRaw_noL_noMissing_revmiRNA.log 2>&1 &

# continue train on selected perfect seed match
nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
    --mRNA_max_length 40 \
    --miRNA_max_length 26 \
    --device cuda:3 \
    --epochs 50 \
    --batch_size 64 \
    --base_model_name TwoTowerMLP \
    --model_name TwoTowerMLP \
    --dataset_name selected_perfect_seed_match \
    --train_dataset_path $data_dir/selected_perfect_seed_match_train.csv \
    --val_dataset_path $data_dir/selected_perfect_seed_match_validation.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
    --accumulation_step 2 \
    --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/miRAW_noL_noMissing/TwoTowerMLP/40/checkpoint_epoch_69.pth \
    > output_logs/output_TwoTowerMLP_selected_perfect_seed_match_revmiRNA.log 2>&1 &

# train on synthetic dataset
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_MLP.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26\
#     --device cuda:2 \
#     --num_epochs 50 \
#     --batch_size 64 \
#     --dataset_name selected_perfect_seed_match \
#     --train_dataset_path $data_dir/selected_perfect_seed_match_train.csv \
#     --val_dataset_path $data_dir/selected_perfect_seed_match_validation.csv \
#     > output_logs/output_MLP_selected_perfect_seed_match.log 2>&1 &