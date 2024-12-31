#!/bin/bash

data_dir=/home/mcb/users/jgu13/projects/mirLM/data

# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 1000 --device cuda:0 --num_epochs 50 > output_logs/output_MLP_1000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 2000 --device cuda:0 --num_epochs 50 > output_logs/output_MLP_2000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 3000 --device cuda:0 --num_epochs 50 > output_logs/output_MLP_3000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 4000 --device cuda:1 --num_epochs 50 > output_logs/output_MLP_4000_50_reverse.log 2>&1 &
# nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py --mRNA_max_len 5000 --device cuda:2 --num_epochs 50 > output_logs/output_MLP_5000_50_reverse.log 2>&1 &

# train on miRAW dataset
nohup python /home/mcb/users/jgu13/projects/mirLM/scripts/main_train_MLP.py \
    --mRNA_max_len 40 \
    --device cuda:2 \
    --num_epochs 1 \
    --batch_size 64 \
    --dataset_name miRAW \
    --dataset_path $data_dir/miRAW_dataset.csv \
    > output_logs/output_MLP_debug.log 2>&1 &

