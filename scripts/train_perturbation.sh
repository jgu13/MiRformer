# train on TargetScan dataset
nohup python scripts/train_perturbation.py \
    --mRNA_max_len 30 \
    --miRNA_max_len 24 \
    --device cuda:2 \
    --epochs 100 \
    --batch_size 32 \
    --base_model_name HyenaDNA \
    --model_name HyenaDNA_w_linker_revmiRNA \
    --dataset_name TargetScan_perturbation \
    --train_dataset_path /home/mcb/users/jgu13/projects/mirLM/TargetScan_dataset/positive_samples.csv \
    --basemodel_cfg /home/mcb/users/jgu13/projects/mirLM/checkpoints/Hyenaconfig.json \
    --use_head \
    --accumulation_step 16 \
    > output_logs/output_HyenaDNA_TargetScan_perturbation.log 2>&1 &