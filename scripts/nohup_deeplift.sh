#!/bin/bash
data_dir=/home/mcb/users/jgu13/projects/mirLM/data

nohup python \
    /home/mcb/users/jgu13/projects/mirLM/scripts/deeplift_analysis.py \
    --mRNA_max_length 40 \
    --miRNA_max_length 26 \
    --device cuda:3 \
    --epochs 100 \
    --batch_size 64 \
    --base_model_name HyenaDNA \
    --model_name HyenaDNA \
    --dataset_name miRAW_noL_noMissing \
    --test_dataset_path $data_dir/ISM_data.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig.json \
    --use_head \
    --accumulation_step 8 \
    --evaluate \
    > deeplift_logs/HyenaDNA_ISM.log 2>&1 &