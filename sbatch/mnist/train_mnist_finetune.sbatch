#!/bin/bash
#SBATCH --job-name=mnist-finetune
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/mnist_finetune_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# MNIST finetune on held-out digit with generative or field replay.
#
#   sbatch --export=ALL,PRETRAIN_RUN=mnist_pretrain_d8,RUN_NAME=mnist_ft_generative_replay_cond_d8, \
#     FINETUNE_STRATEGY=generative_replay sbatch/mnist/train_mnist_finetune.sbatch
#
#   sbatch --export=ALL,PRETRAIN_RUN=mnist_pretrain_uncond_d8,UNCONDITIONAL=1, \
#     RUN_NAME=mnist_ft_generative_field_replay_uncond_d8, \
#     FINETUNE_STRATEGY=generative_field_replay sbatch/mnist/train_mnist_finetune.sbatch
#
# Override: FINETUNE_EPOCHS, GENERATIVE_REPLAY_FRACTION, BATCH_SIZE, HELD_OUT_DIGIT, ...

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${RUN_NAME:=mnist_finetune_${SLURM_JOB_ID}}"
: "${HELD_OUT_DIGIT:=8}"
: "${UNCONDITIONAL:=0}"
: "${FINETUNE_STRATEGY:=generative_field_replay}"

if [[ "${UNCONDITIONAL}" == "1" ]]; then
  : "${PRETRAIN_RUN:=mnist_pretrain_uncond_d8}"
else
  : "${PRETRAIN_RUN:=mnist_pretrain_d8}"
fi

: "${FINETUNE_EPOCHS:=30}"
: "${FINETUNE_LR:=}"
: "${BATCH_SIZE:=256}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cuda}"

: "${FINETUNE_TRAIN_VF:=1}"
: "${FINETUNE_TRAIN_OLD_EMBEDDINGS:=0}"

: "${GENERATIVE_REPLAY_FRACTION:=0.25}"
: "${GENERATIVE_REPLAY_ODE_METHOD:=euler}"
: "${GENERATIVE_REPLAY_ODE_RTOL:=1e-5}"
: "${GENERATIVE_REPLAY_ODE_ATOL:=1e-5}"
: "${GENERATIVE_REPLAY_ODE_STEP_SIZE:=0.05}"

: "${ODE_METHOD:=euler}"
: "${ODE_RTOL:=1e-5}"
: "${ODE_ATOL:=1e-5}"
: "${ODE_STEP_SIZE:=0.05}"
: "${EVAL_MAX_SAMPLES_PER_STRATUM:=64}"
: "${EVAL_UNCOND_N_SAMPLES:=5000}"
: "${EVAL_UNCOND_BATCH_SIZE:=256}"
: "${KNN_K:=5}"
: "${KNN_TRAIN_MAX:=20000}"
: "${PLOT_N_PER_DIGIT:=8}"

case "${FINETUNE_STRATEGY}" in
  generative_replay | generative_field_replay | standard) ;;
  *)
    echo "Invalid FINETUNE_STRATEGY=${FINETUNE_STRATEGY} (expected generative_replay, generative_field_replay, or standard)" >&2
    exit 1
    ;;
esac

extra_uncond=()
if [[ "${UNCONDITIONAL}" == "1" ]]; then
  extra_uncond+=(--unconditional)
fi

finetune_extra=()
case "${FINETUNE_TRAIN_VF,,}" in
  1 | true | yes) finetune_extra+=(--finetune-train-vf) ;;
  0 | false | no) finetune_extra+=(--no-finetune-train-vf) ;;
  *) echo "Invalid FINETUNE_TRAIN_VF=${FINETUNE_TRAIN_VF}" >&2; exit 1 ;;
esac
case "${FINETUNE_TRAIN_OLD_EMBEDDINGS,,}" in
  1 | true | yes) finetune_extra+=(--finetune-train-old-embeddings) ;;
  0 | false | no) finetune_extra+=(--no-finetune-train-old-embeddings) ;;
  *) echo "Invalid FINETUNE_TRAIN_OLD_EMBEDDINGS=${FINETUNE_TRAIN_OLD_EMBEDDINGS}" >&2; exit 1 ;;
esac

extra_ft_lr=()
if [[ -n "${FINETUNE_LR}" ]]; then
  extra_ft_lr+=(--finetune-lr "${FINETUNE_LR}")
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  RUN_NAME=${RUN_NAME}  DEVICE=${DEVICE}  UNCONDITIONAL=${UNCONDITIONAL}"
echo "  PRETRAIN_RUN=${PRETRAIN_RUN}  FINETUNE_STRATEGY=${FINETUNE_STRATEGY}"
echo "  HELD_OUT_DIGIT=${HELD_OUT_DIGIT}  FINETUNE_EPOCHS=${FINETUNE_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}"
echo "  GENERATIVE_REPLAY_FRACTION=${GENERATIVE_REPLAY_FRACTION}"

uv sync
uv run python experiments/mnist/train_mnist_finetune.py \
  --pretrain-run "${PRETRAIN_RUN}" \
  --held-out-digit "${HELD_OUT_DIGIT}" \
  --finetune-strategy "${FINETUNE_STRATEGY}" \
  --finetune-epochs "${FINETUNE_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --generative-replay-fraction "${GENERATIVE_REPLAY_FRACTION}" \
  --generative-replay-ode-method "${GENERATIVE_REPLAY_ODE_METHOD}" \
  --generative-replay-ode-rtol "${GENERATIVE_REPLAY_ODE_RTOL}" \
  --generative-replay-ode-atol "${GENERATIVE_REPLAY_ODE_ATOL}" \
  --generative-replay-ode-step-size "${GENERATIVE_REPLAY_ODE_STEP_SIZE}" \
  --ode-method "${ODE_METHOD}" \
  --ode-rtol "${ODE_RTOL}" \
  --ode-atol "${ODE_ATOL}" \
  --ode-step-size "${ODE_STEP_SIZE}" \
  --eval-max-samples-per-stratum "${EVAL_MAX_SAMPLES_PER_STRATUM}" \
  --eval-uncond-n-samples "${EVAL_UNCOND_N_SAMPLES}" \
  --eval-uncond-batch-size "${EVAL_UNCOND_BATCH_SIZE}" \
  --knn-k "${KNN_K}" \
  --knn-train-max "${KNN_TRAIN_MAX}" \
  --plot-n-per-digit "${PLOT_N_PER_DIGIT}" \
  --run-name "${RUN_NAME}" \
  "${extra_uncond[@]}" \
  "${finetune_extra[@]}" \
  "${extra_ft_lr[@]}"

echo "Finished at $(date -Is)"
echo "Results: $(pwd)/results_dir/${RUN_NAME}/"
