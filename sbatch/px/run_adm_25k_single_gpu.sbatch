#!/usr/bin/env bash
#SBATCH --job-name=px-adm-25k
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=sbatch/logs/px_adm_25k_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${BATCH_SIZE:=32}"
: "${NUM_STEPS:=25000}"
: "${EVAL_EVERY:=5000}"
: "${CHECKPOINT_EVERY:=5000}"
: "${METRICS_EVERY:=25000}"
: "${OUTPUT_DIR:=results_dir/px/adm_25k_single_gpu}"

uv run python experiments/px/main.py \
  --launcher.cluster debug \
  --launcher.output_dir "${OUTPUT_DIR}" \
  --model-name ADM \
  --batch-size "${BATCH_SIZE}" \
  --num-steps "${NUM_STEPS}" \
  --eval-every-n-steps "${EVAL_EVERY}" \
  --checkpoint-every-n-steps "${CHECKPOINT_EVERY}" \
  --metrics-every-n-steps "${METRICS_EVERY}"
