#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nohup python scripts/dna_bert_main.py > dna_bert_logs/train_dna_bert.log 2>&1 &
