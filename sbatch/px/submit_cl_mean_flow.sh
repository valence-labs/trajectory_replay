#!/usr/bin/env bash
# Submit ADM MeanFlow replay jobs for RxRx1 CL.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

STRATEGIES="${STRATEGIES:-generative trajectory}"
MEAN_FLOW_STEPS="${MEAN_FLOW_STEPS:-1 4}"
LAMBDA_MODES="${LAMBDA_MODES:-equal grid}"
SEEDS="${SEEDS:-42}"
STEPS_PER_TASK="${STEPS_PER_TASK:-25000}"
MEAN_FLOW_TRAIN_STEPS="${MEAN_FLOW_TRAIN_STEPS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPUS="${GPUS:-1}"
NODES="${NODES:-1}"
TIMEOUT_MIN="${TIMEOUT_MIN:-2880}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_dir/px/cl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

for strategy in ${STRATEGIES}; do
  for n_steps in ${MEAN_FLOW_STEPS}; do
    for lambda_mode in ${LAMBDA_MODES}; do
      for seed in ${SEEDS}; do
        PYTHONPATH=experiments/px uv run python experiments/px/train_cl.py \
          --launcher.cluster slurm \
          --launcher.nodes "${NODES}" \
          --launcher.gpus "${GPUS}" \
          --launcher.timeout "${TIMEOUT_MIN}" \
          --launcher.output_dir \
            "${OUTPUT_ROOT}/${strategy}_mf_n${n_steps}_${lambda_mode}_s${seed}" \
          --strategy "${strategy}" \
          --model-name ADM \
          --seed "${seed}" \
          --steps-per-task "${STEPS_PER_TASK}" \
          --batch-size "${BATCH_SIZE}" \
          --replay-backend mean_flow \
          --mean-flow-n-steps "${n_steps}" \
          --mean-flow-train-steps "${MEAN_FLOW_TRAIN_STEPS}" \
          --lambda-mode "${lambda_mode}" \
          --lambda-grid 0.25 0.5 0.75
      done
    done
  done
done
