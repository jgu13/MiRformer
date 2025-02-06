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
nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
    --mRNA_max_length 40 \
    --miRNA_max_length 26\
    --device cuda:2 \
    --epochs 50 \
    --batch_size 64 \
    --base_model_name TwoTowerMLP \
    --model_name TwoTower_Hyena_CrossAttn_MLP \
    --dataset_name miRAW_noL_noMissing \
    --train_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_train.csv \
    --val_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_validation.csv \
    > output_logs/output_TwoTowerMLP_miRaw_noL_noMissing_revmiRNA.log 2>&1 &
