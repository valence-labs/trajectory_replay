#!/usr/bin/env bash
# Submit validation-FID selected RxRx1 replay jobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

STRATEGIES="${STRATEGIES:-generative trajectory}"
SEEDS="${SEEDS:-42}"
STEPS_PER_TASK="${STEPS_PER_TASK:-25000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPUS="${GPUS:-1}"
NODES="${NODES:-1}"
TIMEOUT_MIN="${TIMEOUT_MIN:-2880}"
REPLAY_STEP_SIZE="${REPLAY_STEP_SIZE:-0.05}"
LAMBDA_EVAL_STEP_SIZE="${LAMBDA_EVAL_STEP_SIZE:-0.05}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_dir/px/cl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

for strategy in ${STRATEGIES}; do
  for seed in ${SEEDS}; do
    PYTHONPATH=experiments/px uv run python experiments/px/train_cl.py \
      --launcher.cluster slurm \
      --launcher.nodes "${NODES}" \
      --launcher.gpus "${GPUS}" \
      --launcher.timeout "${TIMEOUT_MIN}" \
      --launcher.output_dir "${OUTPUT_ROOT}/${strategy}_grid_s${seed}" \
      --strategy "${strategy}" \
      --model-name ADM \
      --seed "${seed}" \
      --steps-per-task "${STEPS_PER_TASK}" \
      --batch-size "${BATCH_SIZE}" \
      --lambda-mode grid \
      --lambda-grid 0.25 0.5 0.75 \
      --lambda-eval-step-size "${LAMBDA_EVAL_STEP_SIZE}" \
      --replay-step-size "${REPLAY_STEP_SIZE}"
  done
done
