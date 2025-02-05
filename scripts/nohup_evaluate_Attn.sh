# evaluate synthetic dataset
nohup python \
    scripts/main_finetune_Hyena_ddp.py \
    --mRNA_max_len 40 \
    --miRNA_max_len 26 \
    --device cuda:2 \
    --batch_size 64 \
    --model_name Attn \
    --dataset_name selected_perfect_seed_match \
    --test_dataset_path $data_dir/selected_perfect_seed_match_test.csv \
    --evaluate \
    > evaluate_logs/evaluate_Hyena_selected_perfect_seed_match.log 2>&1 &