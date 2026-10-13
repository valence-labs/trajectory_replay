#!/usr/bin/env bash
# Submit the core RxRx1 experiment-incremental ADM campaign.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

STRATEGIES="${STRATEGIES:-naive cumulative real generative trajectory}"
SEEDS="${SEEDS:-42}"
MODEL_NAME="${MODEL_NAME:-ADM}"
STEPS_PER_TASK="${STEPS_PER_TASK:-25000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPUS="${GPUS:-1}"
NODES="${NODES:-1}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1440}"
REPLAY_STEP_SIZE="${REPLAY_STEP_SIZE:-0.05}"
EVAL_FID="${EVAL_FID:-1}"
NO_SNAPSHOT="${NO_SNAPSHOT:-0}"
NO_COMPILE="${NO_COMPILE:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_dir/px/cl}"
export WANDB_MODE

for strategy in ${STRATEGIES}; do
  for seed in ${SEEDS}; do
    extra=()
    if [[ "${EVAL_FID}" == "1" ]]; then
      extra+=(--eval-fid)
    else
      extra+=(--no-eval-fid)
    fi
    if [[ "${NO_SNAPSHOT}" == "1" ]]; then
      extra+=(--launcher.no_snapshot)
    fi
    if [[ "${NO_COMPILE}" == "1" ]]; then
      extra+=(--no-compile)
    fi
    echo "Submitting strategy=${strategy} seed=${seed}"
    PYTHONPATH=experiments/px uv run python experiments/px/train_cl.py \
      --launcher.cluster slurm \
      --launcher.nodes "${NODES}" \
      --launcher.gpus "${GPUS}" \
      --launcher.timeout "${TIMEOUT_MIN}" \
      --launcher.output_dir "${OUTPUT_ROOT}/${strategy}_equal_s${seed}" \
      --strategy "${strategy}" \
      --model-name "${MODEL_NAME}" \
      --seed "${seed}" \
      --steps-per-task "${STEPS_PER_TASK}" \
      --batch-size "${BATCH_SIZE}" \
      --lambda-mode equal \
      --replay-step-size "${REPLAY_STEP_SIZE}" \
      "${extra[@]}"
  done
done
