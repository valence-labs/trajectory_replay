#!/bin/bash
#SBATCH --job-name=mnist-seen-plots
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/mnist_seen_plots_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Regenerate real_p*/gen_p* grids for MNIST finetune checkpoints (GPU only).
# Conditional: per-digit sampling. Unconditional: KNN-stratified uncond samples.
#
#   sbatch --export=ALL,RUN_NAME=mnist/mnist_ft_generative_replay_cond_d8 \
#     sbatch/mnist/regenerate_seen_digit_plots.sbatch
#
#   sbatch --export=ALL,RUN_NAME=mnist/mnist_ft_generative_replay_uncond_r75_d8 \
#     sbatch/mnist/regenerate_seen_digit_plots.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${RUN_NAME:?Set RUN_NAME to a run under results_dir/}"
: "${DEVICE:=cuda}"
: "${PLOT_N_PER_DIGIT:=8}"
: "${MAX_DIGITS:=3}"
: "${EVAL_SEED:=0}"
: "${PLOT_UNCOND_N_SAMPLES:=3000}"
: "${KNN_K:=5}"
: "${KNN_TRAIN_MAX:=20000}"
: "${EVAL_UNCOND_BATCH_SIZE:=256}"

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "  RUN_NAME=${RUN_NAME}  DEVICE=${DEVICE}"

uv sync
uv run python experiments/mnist/regenerate_seen_digit_plots.py \
  --run-name "${RUN_NAME}" \
  --device "${DEVICE}" \
  --plot-n-per-digit "${PLOT_N_PER_DIGIT}" \
  --max-digits "${MAX_DIGITS}" \
  --eval-seed "${EVAL_SEED}" \
  --plot-uncond-n-samples "${PLOT_UNCOND_N_SAMPLES}" \
  --knn-k "${KNN_K}" \
  --knn-train-max "${KNN_TRAIN_MAX}" \
  --eval-uncond-batch-size "${EVAL_UNCOND_BATCH_SIZE}"

echo "Finished at $(date -Is)"
echo "Results: $(pwd)/results_dir/${RUN_NAME}"
