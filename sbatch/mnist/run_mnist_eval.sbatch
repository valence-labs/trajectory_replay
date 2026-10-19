#!/bin/bash
#SBATCH --job-name=mnist-eval
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/mnist_eval_%j.log
#SBATCH --gres=gpu:1

# Re-run forget/fit eval for an existing MNIST checkpoint.
#
#   sbatch --export=ALL,RUN_NAME=mnist_ft_generative_replay_uncond_d8 \
#     sbatch/mnist/run_mnist_eval.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${RUN_NAME:?Set RUN_NAME}"
: "${DEVICE:=cuda}"
: "${EVAL_SEED:=0}"
: "${EVAL_MAX_SAMPLES_PER_STRATUM:=64}"
: "${EVAL_UNCOND_N_SAMPLES:=5000}"
: "${EVAL_UNCOND_BATCH_SIZE:=256}"
: "${KNN_K:=5}"
: "${KNN_TRAIN_MAX:=20000}"

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "  RUN_NAME=${RUN_NAME}  DEVICE=${DEVICE}"

uv sync
uv run python experiments/mnist/run_mnist_eval.py \
  --run-name "${RUN_NAME}" \
  --device "${DEVICE}" \
  --eval-seed "${EVAL_SEED}" \
  --eval-max-samples-per-stratum "${EVAL_MAX_SAMPLES_PER_STRATUM}" \
  --eval-uncond-n-samples "${EVAL_UNCOND_N_SAMPLES}" \
  --eval-uncond-batch-size "${EVAL_UNCOND_BATCH_SIZE}" \
  --knn-k "${KNN_K}" \
  --knn-train-max "${KNN_TRAIN_MAX}"

echo "Finished at $(date -Is)"
