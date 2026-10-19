#!/usr/bin/env bash
#SBATCH --job-name=px-eval-task0
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/px_eval_task0_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${SEED:=42}"
: "${RESULTS_ROOT:=results_dir/px/cl_final}"

export PYTHONPATH="${REPO_ROOT}/experiments/px${PYTHONPATH:+:${PYTHONPATH}}"

uv run python experiments/px/eval_task0_checkpoint.py \
  --checkpoint "${RESULTS_ROOT}/task0_s${SEED}/checkpoint.ckpt" \
  --num-workers 8
