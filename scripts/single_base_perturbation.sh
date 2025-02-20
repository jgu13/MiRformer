nohup python \
    scripts/single_base_perturbation.py \
    --mRNA_max_length 30 \
    --miRNA_max_length 24 \
    --device cuda:2 \
    --base_model_name HyenaDNA \
    --model_name HyenaDNA_w_linker_revmiRNA \
    --dataset_name TargetScan \
    --test_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/TargetScan_validation.csv \
    --use_head \
    --save_plot_dir /home/mcb/users/jgu13/projects/mirLM/Performance/TargetScan_test/viz_seq_perturbation/7 \
    > single_base_perturbation.out 2>&1 &