#!/bin/bash

# Launch script for DDP training
# Usage: bash launch_ddp_training.sh [options]

# Default parameters
EPOCHS=1
BATCH_SIZE=32
ACCUMULATION_STEP=8
EMBED_DIM=1024
FF_DIM=4096
LR=1e-4
MRNA_MAX_LEN=520
MIRNA_MAX_LEN=24

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --accumulation_step)
            ACCUMULATION_STEP="$2"
            shift 2
            ;;
        --embed_dim)
            EMBED_DIM="$2"
            shift 2
            ;;
        --ff_dim)
            FF_DIM="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --mrna_max_len)
            MRNA_MAX_LEN="$2"
            shift 2
            ;;
        --mirna_max_len)
            MIRNA_MAX_LEN="$2"
            shift 2
            ;;
        --world_size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --epochs EPOCHS              Number of epochs (default: $EPOCHS)"
            echo "  --batch_size BATCH_SIZE      Batch size per GPU (default: $BATCH_SIZE)"
            echo "  --accumulation_step STEPS    Gradient accumulation steps (default: $ACCUMULATION_STEP)"
            echo "  --embed_dim DIM              Embedding dimension (default: $EMBED_DIM)"
            echo "  --ff_dim DIM                 Feed-forward dimension (default: $FF_DIM)"
            echo "  --lr LR                      Learning rate (default: $LR)"
            echo "  --mrna_max_len LEN           Maximum mRNA length (default: $MRNA_MAX_LEN)"
            echo "  --mirna_max_len LEN          Maximum miRNA length (default: $MIRNA_MAX_LEN)"
            echo "  --world_size SIZE            Number of GPUs (default: auto-detect)"
            echo "  -h, --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    echo "Error: CUDA is not available. DDP requires CUDA."
    exit 1
fi

# Get number of available GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ -n "$WORLD_SIZE" ]]; then
    NUM_GPUS=$WORLD_SIZE
fi

echo "Starting DDP training with $NUM_GPUS GPUs"
echo "Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Accumulation steps: $ACCUMULATION_STEP"
echo "  Embedding dimension: $EMBED_DIM"
echo "  Feed-forward dimension: $FF_DIM"
echo "  Learning rate: $LR"
echo "  mRNA max length: $MRNA_MAX_LEN"
echo "  miRNA max length: $MIRNA_MAX_LEN"
echo "  World size: $NUM_GPUS"

# Launch the training
python scripts/Pretrain_DDP.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulation_step $ACCUMULATION_STEP \
    --embed_dim $EMBED_DIM \
    --ff_dim $FF_DIM \
    --lr $LR \
    --mrna_max_len $MRNA_MAX_LEN \
    --mirna_max_len $MIRNA_MAX_LEN \
    --world_size $NUM_GPUS
