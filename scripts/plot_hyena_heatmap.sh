#!/bin/bash

nohup python \
    scripts/plot_heatmap.py \
    --mRNA_max_length 30 \
    --miRNA_max_length 24 \
    --device cuda:3 \
    --base_model_name TwoTowerMLP \
    --model_name TwoTowerMLP_30 \
    --dataset_name TargetScan \
    --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
    --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan/TwoTowerMLP_30/ \
    > plot_heatmap.out 2>&1 &