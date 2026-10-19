#!/bin/bash
#SBATCH --job-name=mnist-pretrain
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=sbatch/logs/mnist_pretrain_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# MNIST pretrain (conditional or unconditional).
#
#   sbatch --export=ALL,RUN_NAME=mnist_pretrain_d8 sbatch/mnist/train_mnist_pretrain.sbatch
#   sbatch --export=ALL,RUN_NAME=mnist_pretrain_uncond_d8,UNCONDITIONAL=1 sbatch/mnist/train_mnist_pretrain.sbatch
#
# Override: EPOCHS, BATCH_SIZE, LR, HELD_OUT_DIGIT, DEVICE, ...

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${RUN_NAME:=mnist_pretrain_${SLURM_JOB_ID}}"
: "${HELD_OUT_DIGIT:=8}"
: "${UNCONDITIONAL:=0}"

: "${BACKBONE_TYPE:=unet}"
: "${COND_EMB_DIM:=32}"
: "${TIME_EMB_DIM:=64}"
: "${HIDDEN_DIM:=256}"
: "${DEPTH:=4}"
: "${UNET_NUM_CHANNELS:=64}"
: "${UNET_NUM_RES_BLOCKS:=2}"
: "${UNET_ATTENTION_RESOLUTIONS:=14}"
: "${UNET_DROPOUT:=0.0}"

: "${EPOCHS:=100}"
: "${BATCH_SIZE:=256}"
: "${LR:=2e-4}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cuda}"
: "${MAX_SAMPLES:=}"

: "${ODE_METHOD:=euler}"
: "${ODE_RTOL:=1e-5}"
: "${ODE_ATOL:=1e-5}"
: "${ODE_STEP_SIZE:=0.05}"
: "${EVAL_MAX_SAMPLES_PER_STRATUM:=64}"
: "${EVAL_UNCOND_N_SAMPLES:=5000}"
: "${EVAL_UNCOND_BATCH_SIZE:=256}"
: "${KNN_K:=5}"
: "${KNN_TRAIN_MAX:=20000}"

extra_uncond=()
if [[ "${UNCONDITIONAL}" == "1" ]]; then
  extra_uncond+=(--unconditional)
fi

extra_max=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  extra_max+=(--max-samples "${MAX_SAMPLES}")
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  RUN_NAME=${RUN_NAME}  DEVICE=${DEVICE}  UNCONDITIONAL=${UNCONDITIONAL}"
echo "  HELD_OUT_DIGIT=${HELD_OUT_DIGIT}  EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}"

uv sync
uv run python experiments/mnist/train_mnist_pretrain.py \
  --held-out-digit "${HELD_OUT_DIGIT}" \
  --backbone-type "${BACKBONE_TYPE}" \
  --cond-emb-dim "${COND_EMB_DIM}" \
  --time-emb-dim "${TIME_EMB_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --unet-num-channels "${UNET_NUM_CHANNELS}" \
  --unet-num-res-blocks "${UNET_NUM_RES_BLOCKS}" \
  --unet-attention-resolutions "${UNET_ATTENTION_RESOLUTIONS}" \
  --unet-dropout "${UNET_DROPOUT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --ode-method "${ODE_METHOD}" \
  --ode-rtol "${ODE_RTOL}" \
  --ode-atol "${ODE_ATOL}" \
  --ode-step-size "${ODE_STEP_SIZE}" \
  --eval-max-samples-per-stratum "${EVAL_MAX_SAMPLES_PER_STRATUM}" \
  --eval-uncond-n-samples "${EVAL_UNCOND_N_SAMPLES}" \
  --eval-uncond-batch-size "${EVAL_UNCOND_BATCH_SIZE}" \
  --knn-k "${KNN_K}" \
  --knn-train-max "${KNN_TRAIN_MAX}" \
  --run-name "${RUN_NAME}" \
  "${extra_uncond[@]}" \
  "${extra_max[@]}"

echo "Finished at $(date -Is)"
echo "Results: $(pwd)/results_dir/${RUN_NAME}"
