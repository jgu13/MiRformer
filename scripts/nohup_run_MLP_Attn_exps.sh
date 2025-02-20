#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# train on ISM data
# nohup python \
#     /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
#     --mRNA_max_length 40 \
#     --miRNA_max_length 26 \
#     --device cuda:2 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name MLP_Attn \
#     --dataset_name miRAW_noL_noMissing \
#     --train_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_train.csv \
#     --val_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_validation.csv \
#     --backbone_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/hyenadna-small-32k-seqlen/attn_config.json \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLP_Attnconfig.json \
#     --accumulation_step 2 \
#     > output_logs/output_MLPAttn_miRaw_noL_noMissing.log 2>&1 &

# continue train on selected perfect seed match
nohup python \
    /home/mcb/users/jgu13/projects/mirLM/scripts/main.py \
    --mRNA_max_length 40 \
    --miRNA_max_length 26 \
    --device cuda:2 \
    --epochs 30 \
    --batch_size 64 \
    --base_model_name TwoTowerMLP \
    --model_name MLP_Attn \
    --dataset_name selected_perfect_seed_match \
    --train_dataset_path $data_dir/selected_perfect_seed_match_train.csv \
    --val_dataset_path $data_dir/selected_perfect_seed_match_validation.csv \
    --backbone_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/hyenadna-small-32k-seqlen/attn_config.json \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLP_Attnconfig.json \
    --accumulation_step 2 \
    --resume_ckpt /home/mcb/users/jgu13/projects/mirLM/checkpoints/miRAW_noL_noMissing/MLP_Attn/40/checkpoint_epoch_9.pth \
    > output_logs/output_MLPAttn_selected_perfect_seed_match.log 2>&1 &