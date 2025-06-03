#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# train HyenaDNA with linker and reverse miRNA on miRAW dataset
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
#     --mRNA_max_len 40 \
#     --miRNA_max_len 26\
#     --device cuda:2 \
#     --epochs 50 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_w_linker_revmiRNA \
#     --dataset_name miRAW_noL_noMissing \
#     --train_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_train.csv \
#     --val_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_validation.csv \
#     --use_head \
#     > output_logs/output_HyenaDNA_miRaw_noL_noMissing_w_linker_revmiRNA.log 2>&1 &

# train TwoTowerMLP on miRAW dataset
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
#     --mRNA_max_length 40 \
#     --miRNA_max_length 26\
#     --device cuda:2 \
#     --epochs 50 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTower_Hyena_CrossAttn_MLP \
#     --dataset_name miRAW_noL_noMissing \
#     --train_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_train.csv \
#     --val_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_validation.csv \
#     > output_logs/output_TwoTowerMLP_miRaw_noL_noMissing_revmiRNA.log 2>&1 &

# train TwoTowerMLP on TargetScan dataset
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
#     --mRNA_max_length 30 \
#     --miRNA_max_length 24 \
#     --device cuda:2 \
#     --epochs 200 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTowerMLP_30 \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
#     > output_logs/output_TwoTowerMLP_TargetScan_30.log 2>&1 &

# train HyenaDNA on TargetScan dataset
nohup python scripts/main.py \
    --mRNA_max_len 30 \
    --miRNA_max_len 24 \
    --device cuda:3 \
    --epochs 300 \
    --batch_size 16 \
    --base_model_name HyenaDNA \
    --model_name HyenaDNA_miRNA_few_shot \
    --dataset_name TargetScan \
    --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_few_shot_train.csv \
    --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig_30.json \
    --use_head \
    > output_logs/output_HyenaDNA_TargetScan_30_few_shot.log 2>&1 &

# nohup python scripts/main.py \
#     --mRNA_max_len 500 \
#     --miRNA_max_len 24 \
#     --device cuda:3 \
#     --epochs 200 \
#     --batch_size 32 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_500_perturbed \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_500.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_500.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig_500.json \
#     --use_head \
#     > output_logs/output_HyenaDNA_TargetScan_500_perturbed.log 2>&1 &

# nohup python scripts/main.py \
#     --mRNA_max_len 924 \
#     --device cuda:1 \
#     --epochs 200 \
#     --batch_size 2 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_924_perturbed \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_924.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_924.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig_924.json \
#     --use_head \
#     > output_logs/output_HyenaDNA_TargetScan_924_perturbed.log 2>&1 &

# nohup python scripts/main.py \
#     --mRNA_max_len 10000 \
#     --miRNA_max_len 24 \
#     --device cuda:2 \
#     --epochs 200 \
#     --batch_size 1 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_10k_perturbed \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_10k.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_10k.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig_10k.json \
#     --use_head \
#     > output_logs/output_HyenaDNA_TargetScan_10k_perturbed.log 2>&1 &

# nohup python scripts/main.py \
#     --mRNA_max_len 22552 \
#     --miRNA_max_len 24 \
#     --device cuda:0 \
#     --epochs 200 \
#     --batch_size 4 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_miRNA \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_10k.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_10k.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig_10k.json \
#     --use_head \
#     > output_logs/output_HyenaDNA_TargetScan_full_length_perturbed.log 2>&1 &

# train two-tower on TargetScan
# nohup python scripts/main.py \
#     --mRNA_max_len 500 \
#     --miRNA_max_len 24 \
#     --device cuda:3 \
#     --epochs 200 \
#     --batch_size 32 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTowerMLP_500_perturbed \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_500.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_500.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig_500.json \
#     > output_logs/output_TwoTowerMLP_TargetScan_500_perturbed.log 2>&1 &

# nohup python scripts/main.py \
#     --mRNA_max_len 924 \
#     --device cuda:2 \
#     --epochs 200 \
#     --batch_size 2 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTowerMLP_924_perturbed \
#     --dataset_name TargetScan \
#     --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_924.csv \
#     --val_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation_924.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig_924.json \
#     > output_logs/output_TwoTowerMLP_TargetScan_924_perturbed.log 2>&1 &