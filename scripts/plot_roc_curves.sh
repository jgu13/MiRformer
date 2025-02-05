#!/bin/bash

mirtar_clip_dir=/home/mcb/users/jgu13/projects/mirLM/Performance/DeepMirTar-par-clip-selected
# python scripts/plot_roc_curves.py \
#     --json_files $mirtar_clip_dir/MLP/predictions_1000.json \
#                 $mirtar_clip_dir/HyenaDNA/predictions_1000.json \
#                 $mirtar_clip_dir/CNN/predictions_1000.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file $mirtar_clip_dir/1000_ROC.png

# python scripts/plot_roc_curves.py \
#     --json_files $mirtar_clip_dir/MLP/predictions_2000.json \
#                 $mirtar_clip_dir/HyenaDNA/predictions_2000.json \
#                 $mirtar_clip_dir/CNN/predictions_2000.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file $mirtar_clip_dir/2000_ROC.png

# python scripts/plot_roc_curves.py \
#     --json_files $mirtar_clip_dir/MLP/predictions_3000.json \
#                 $mirtar_clip_dir/HyenaDNA/predictions_3000.json \
#                 $mirtar_clip_dir/CNN/predictions_3000.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file $mirtar_clip_dir/3000_ROC.png

# python scripts/plot_roc_curves.py \
#     --json_files $mirtar_clip_dir/MLP/predictions_4000.json \
#                 $mirtar_clip_dir/HyenaDNA/predictions_4000.json \
#                 $mirtar_clip_dir/CNN/predictions_4000.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file $mirtar_clip_dir/4000_ROC.png

# python scripts/plot_roc_curves.py \
#     --json_files $mirtar_clip_dir/MLP/predictions_5000.json \
#                 $mirtar_clip_dir/HyenaDNA/predictions_5000.json \
#                 $mirtar_clip_dir/CNN/predictions_5000.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file $mirtar_clip_dir/5000_ROC.png

# # plot miraw-trained model performance
# miRAW_noL_noMissing=/home/mcb/users/jgu13/projects/mirLM/Performance/data_miRaw_noL_noMissing_remained_seed1122_test
# python scripts/plot_roc_curves.py \
#     --json_files $miRAW_noL_noMissing/MLP/predictions_40.json \
#                 $miRAW_noL_noMissing/HyenaDNA/predictions_40.json \
#                 $miRAW_noL_noMissing/CNN/predictions_40.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file $miRAW_noL_noMissing/miraw_noL_noMissing_ROC.png

# plot miraw-completely-random trained model performance
# miraw_like_completely_random=/home/mcb/users/jgu13/projects/mirLM/Performance/miRAW_dataset_completely_random_test
# python scripts/plot_roc_curves.py \
#     --json_files $miraw_like_completely_random/MLP/predictions_40.json \
#                 $miraw_like_completely_random/HyenaDNA/predictions_40.json \
#                 $miraw_like_completely_random/CNN/predictions_40.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN-MLP \
#     --output_file $miraw_like_completely_random/miraw_random_ROC.png

# par-clip data
# miraw_like_completely_random=/home/mcb/users/jgu13/projects/mirLM/Performance/DeepMirTar-par-clip-miraw-like-completely-random
# python scripts/plot_roc_curves.py \
#     --json_files $miraw_like_completely_random/MLP/predictions_40.json \
#                 $miraw_like_completely_random/HyenaDNA/predictions_40.json \
#                 $miraw_like_completely_random/CNN/predictions_40.json \
#     --method_names Two-tower-HyenaDNA-MLP-CrossAttn Concate-Finetuned-HyenaDNA-MLP baseline-CNN-MLP \
#     --output_file $miraw_like_completely_random/miraw_random_ROC.png

# miraw_like_completely_random=/home/mcb/users/jgu13/projects/mirLM/Performance/miRAW_dataset_completely_random_validation
# python scripts/plot_roc_curves.py \
#     --json_files $miraw_like_completely_random/CNN/predictions_40.json \
#     --method_names baseline-CNN-MLP \
#     --output_file $miraw_like_completely_random/miraw_random_ROC.png

# plot perfect seed match model performance
# syn_data=/home/mcb/users/jgu13/projects/mirLM/Performance/selected_perfect_seed_match_test
# python scripts/plot_roc_curves.py \
#     --json_files $syn_data/MLP/predictions_40.json \
#                 $syn_data/HyenaDNA/predictions_40.json \
#                 $syn_data/CNN/predictions_40.json \
#     --method_names TwoTower-HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP Baseline-CNN \
#     --output_file $syn_data/selected_perfect_seed_match_ROC.png

python scripts/plot_roc_curves.py \
    --json_files /home/mcb/users/jgu13/projects/mirLM/Performance/ISM_data_mutant_mRNA_seed/CNN/predictions_40.json \
                /home/mcb/users/jgu13/projects/mirLM/Performance/ISM_data_mutant_mRNA_nonseed/CNN/predictions_40.json \
                /home/mcb/users/jgu13/projects/mirLM/Performance/ISM_data_mutant_miRNA_seed/CNN/predictions_40.json \
                /home/mcb/users/jgu13/projects/mirLM/Performance/ISM_data_mutant_miRNA_nonseed/CNN/predictions_40.json \
    --method_names ISM-mRNA-seed ISM-mRNA-nonseed ISM-miRNA-seed ISM-miRNA-nonseed \
    --plot_title "CNN on Classifying In-Silico Pertubed mRNA or miRNA pairs" \
    --output_file /home/mcb/users/jgu13/projects/mirLM/Performance/CNN_ISM_ROC.png
