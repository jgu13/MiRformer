# nohup python \
#     scripts/single_base_perturbation.py \
#     --mRNA_max_length 500 \
#     --miRNA_max_length 24 \
#     --device cuda:3 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_500_perturbed \
#     --dataset_name TargetScan \
#     --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_500.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig_500.json \
#     --use_head \
#     --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan_test/viz_seq_perturbation/HyenaDNA_miRNA/500/w_perturb/ \
#     > single_base_perturbation.out 2>&1 &

# nohup python \
#     scripts/single_base_perturbation.py \
#     --mRNA_max_length 924 \
#     --miRNA_max_length 24 \
#     --device cuda:3 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_924_perturbed \
#     --dataset_name TargetScan \
#     --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_924.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig_924.json \
#     --use_head \
#     --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan_test/viz_seq_perturbation/HyenaDNA_miRNA/924/w_perturb/ \
#     > single_base_perturbation.out 2>&1 &

# nohup python \
#     scripts/single_base_perturbation.py \
#     --mRNA_max_length 500 \
#     --miRNA_max_length 24 \
#     --device cuda:3 \
#     --batch_size 64 \
#     --base_model_name TwoTowerMLP \
#     --model_name TwoTowerMLP_500_perturbed \
#     --dataset_name TargetScan \
#     --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_500.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig_500.json \
#     --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan_test/viz_seq_perturbation/TwoTowerMLP/500/w_perturb/ \
#     > single_base_perturbation.out 2>&1 &

# TwoTowerTransformer
nohup python \
    scripts/single_base_perturbation.py \
    --mRNA_max_length 30 \
    --miRNA_max_length 24 \
    --device cuda:2 \
    --batch_size 32 \
    --base_model_name TwoTowerTransformer \
    --model_name TwoTowerTransformer \
    --dataset_name TargetScan \
    --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train_30_randomized_start.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig_500.json \
    --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan_test/viz_seq_perturbation/TwoTowerTransformer/30 \
    > single_base_perturbation.out 2>&1 &