# nohup python \
#     scripts/single_base_perturbation.py \
#     --mRNA_max_length 30 \
#     --miRNA_max_length 24 \
#     --device cuda:3 \
#     --batch_size 64 \
#     --base_model_name HyenaDNA \
#     --model_name HyenaDNA_miRNA \
#     --dataset_name TargetScan \
#     --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train.csv \
#     --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig2.json \
#     --use_head \
#     --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan_test/viz_seq_perturbation/HyenaDNA_miRNA/30/wo_perturb/1 \
#     > single_base_perturbation.out 2>&1 &

nohup python \
    scripts/single_base_perturbation.py \
    --mRNA_max_length 30 \
    --miRNA_max_length 24 \
    --device cuda:3 \
    --batch_size 64 \
    --base_model_name TwoTowerMLP \
    --model_name TwoTowerMLP_30 \
    --dataset_name TargetScan \
    --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_train.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/MLPconfig.json \
    --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan_test/viz_seq_perturbation/TwoTowerMLP/30/wo_perturb/1 \
    > single_base_perturbation.out 2>&1 &