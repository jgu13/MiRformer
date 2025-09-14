#!/bin/bash
#SBATCH --job-name=cls_only
#SBATCH --account=def-liyue
#SBATCH --time=28:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=transformer_500_randomized_start_random_samples_%j.out
#SBATCH --error=transformer_500_randomized_start_random_samples_%j.err

set -euo pipefail

module purge

echo "Starting job at: $(date)"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# Show the GPU Slurm bound to your job
nvidia-smi || true

# Activate conda
source /home/claris/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate mirLM

# Keep thread counts sane for 5 CPUs
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5

# Quick PyTorch CUDA sanity check
python - <<'PY'
import os, torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count(), "name0:", torch.cuda.get_device_name(0))
PY

export WANDB_MODE=offline                 # hard offline (no network calls)
export WANDB_DIR=/home/claris/projects/ctb-liyue/claris/projects/mirLM/wandb
export WANDB_CACHE_DIR=$WANDB_DIR/cache   # optional, keeps cache off $HOME
export WANDB_CONSOLE=off                  # quieter logs
export WANDB_DISABLE_GIT=true             # avoid git calls on Lustre
export WANDB__SERVICE_WAIT=60             # don't wait for background service
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Go to your project dir
cd /home/claris/projects/ctb-liyue/claris/projects/mirLM
echo "PWD: $(pwd)"

# -------- Logging setup --------
LOG="train_TargetScan_500_randomized_start_random_samples_CLS_only.log"
# Run training in the foreground and mirror output to the log file
srun --unbuffered python -u scripts/transformer_model.py 2>&1 | tee "$LOG"



