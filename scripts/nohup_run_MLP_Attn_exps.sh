#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# train on ISM data
nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_MLP.py \
    --mRNA_max_len 40 \
    --miRNA_max_len 26 \
    --device cuda:2 \
    --num_epochs 50 \
    --batch_size 64 \
    --model_name MLP_Attn \
    --dataset_name selected_perfect_seed_match \
    --train_dataset_path $data_dir/selected_perfect_seed_match_train.csv \
    --val_dataset_path $data_dir/selected_perfect_seed_match_validation.csv \
    --backbone_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/hyenadna-small-32k-seqlen/attn_config.json \
    > output_logs/output_MLPAttn_selected_perfect_seed_match.log 2>&1 &
