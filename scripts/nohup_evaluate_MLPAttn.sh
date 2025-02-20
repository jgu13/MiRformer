#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data
# evaluate on miRaw dataset
# nohup python \
#     scripts/main.py \
#     --mRNA_max_length 40 \
#     --miRNA_max_length 26 \
#     --device cuda:2 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name MLP_Attn \
#     --dataset_name miRAW_noL_noMissing \
#     --test_dataset_path $data_dir/data_miRaw_noL_noMissing_remained_seed1122_test.csv \
#     --backbone_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/hyenadna-small-32k-seqlen/attn_config.json \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLP_Attnconfig.json \
#     --accumulation_step 2 \
#     --evaluate \
#     > evaluate_logs/evaluate_MLPAttn_miRaw_noL_noMissing.log 2>&1 &

# evaluate on selected perfect seed match
# nohup python \
#     scripts/main.py \
#     --mRNA_max_length 40 \
#     --miRNA_max_length 26 \
#     --device cuda:2 \
#     --epochs 100 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name MLP_Attn \
#     --dataset_name selected_perfect_seed_match \
#     --test_dataset_path $data_dir/selected_perfect_seed_match_test.csv \
#     --backbone_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/hyenadna-small-32k-seqlen/attn_config.json \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLP_Attnconfig.json \
#     --accumulation_step 2 \
#     --evaluate \
#     > evaluate_logs/evaluate_MLPAttn_perfect_seed_match.log 2>&1 &

# evaluate ISM data
nohup python \
    scripts/main.py \
    --mRNA_max_length 40 \
    --miRNA_max_length 26 \
    --device cuda:2 \
    --epochs 100 \
    --batch_size 64 \
    --base_model_name TwoTowerMLP \
    --model_name MLP_Attn \
    --dataset_name miRAW_noL_noMissing \
    --test_dataset_path $data_dir/ISM_data_mutant_miRNA_seed.csv \
    --backbone_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/hyenadna-small-32k-seqlen/attn_config.json \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLP_Attnconfig.json \
    --accumulation_step 2 \
    --evaluate \
    > evaluate_logs/evaluate_MLPAttn_ISM_miRNA_seed.log 2>&1 &