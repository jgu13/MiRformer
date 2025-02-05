#!/bin/bash

# miRAW_noL_noMissing=/home/mcb/users/jgu13/projects/mirLM/Performance/miRAW_noL_noMissing
# python scripts/plot_model_performance_line_plot.py \
#     --mRNA_max_len 40\
#     --dataset_name miRAW_noL_noMissing\
#     --model_dirs MLP HyenaDNA CNN\
#     --model_names_in_plot TwoTower-HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP Baseline-CNN \
#     --train_loss_save_path $miRAW_noL_noMissing/miraw_noL_noMissing_train_loss.png \
#     --test_acc_save_path $miRAW_noL_noMissing/miraw_noL_noMissing_test_acc.png

syn_data=/home/mcb/users/jgu13/projects/mirLM/Performance/perfect_seed_match
python scripts/plot_model_performance_line_plot.py \
    --mRNA_max_len 40\
    --dataset_name perfect_seed_match\
    --model_dirs MLP HyenaDNA CNN\
    --model_names_in_plot TwoTower-HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP Baseline-CNN \
    --train_loss_save_path $syn_data/perfect_seed_match.png \
    --test_acc_save_path $syn_data/miraw_noL_noMissing_test_acc.png