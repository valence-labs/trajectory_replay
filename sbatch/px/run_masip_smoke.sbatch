#!/usr/bin/env bash
#SBATCH --job-name=px-masip-smoke
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/px_masip_smoke_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${STRATEGY:=generative_distillation}"
: "${MAX_TASKS:=2}"

export WANDB_MODE=offline
uv run python experiments/px/train_masip_cl.py \
  --launcher.cluster debug \
  --launcher.output_dir "results_dir/px/masip_smoke/${STRATEGY}" \
  --strategy "${STRATEGY}" \
  --steps-per-task 1 \
  --batch-size 2 \
  --num-workers 2 \
  --max-tasks "${MAX_TASKS}" \
  --teacher-steps 2 \
  --eval-steps 2 \
  --no-eval-fid \
  --no-compile
