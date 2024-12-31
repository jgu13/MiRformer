#!/bin/bash

mirtar_clip_dir=/home/mcb/users/jgu13/projects/mirLM/Performance/DeepMirTar-par-clip-selected
# python scripts/plot_roc_curves.py \
#     --json_files $mirtar_clip_dir/MLP/predictions_1000.json \
#                 $mirtar_clip_dir/HyenaDNA/predictions_1000.json \
#                 $mirtar_clip_dir/CNN/predictions_1000.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file $mirtar_clip_dir/1000_ROC.png

python scripts/plot_roc_curves.py \
    --json_files $mirtar_clip_dir/MLP/predictions_2000.json \
                $mirtar_clip_dir/HyenaDNA/predictions_2000.json \
                $mirtar_clip_dir/CNN/predictions_2000.json \
    --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
    --output_file $mirtar_clip_dir/2000_ROC.png

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

# plot miraw-trained model performance
# python scripts/plot_roc_curves.py \
#     --json_files /home/mcb/users/jgu13/projects/mirLM/Performance/DeepMirTar-par-clip-miraw-like/MLP/predictions_40.json \
#                 /home/mcb/users/jgu13/projects/mirLM/Performance/DeepMirTar-par-clip-miraw-like/HyenaDNA/predictions_40.json \
#                 /home/mcb/users/jgu13/projects/mirLM/Performance/DeepMirTar-par-clip-miraw-like/CNN/predictions_40.json \
#     --method_names HyenaDNA-MLP-CrossAttn Finetuned-HyenaDNA-MLP baseline-CNN \
#     --output_file /home/mcb/users/jgu13/projects/mirLM/Performance/DeepMirTar-par-clip-miraw-like/miraw_ROC.png
