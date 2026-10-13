#!/usr/bin/env bash
#SBATCH --job-name=px-cl-smoke
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/px_cl_smoke_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${STRATEGY:=generative}"
: "${STEPS_PER_TASK:=1}"
: "${BATCH_SIZE:=2}"
: "${MAX_TASKS:=3}"
: "${REPLAY_STEP_SIZE:=0.5}"
: "${LAMBDA_MODE:=equal}"
: "${LAMBDA_GRID:=0.25 0.5 0.75}"
: "${LAMBDA_EVAL_STEP_SIZE:=0.5}"
: "${REPLAY_BACKEND:=ode}"
: "${MEAN_FLOW_N_STEPS:=1}"
: "${MEAN_FLOW_TRAIN_STEPS:=1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO_ROOT}/experiments/px${PYTHONPATH:+:${PYTHONPATH}}"

uv run python experiments/px/train_cl.py \
  --launcher.cluster debug \
  --launcher.output_dir "results_dir/px/cl_compute_smoke/${STRATEGY}" \
  --strategy "${STRATEGY}" \
  --model-name ADM \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 2 \
  --steps-per-task "${STEPS_PER_TASK}" \
  --max-tasks "${MAX_TASKS}" \
  --replay-step-size "${REPLAY_STEP_SIZE}" \
  --lambda-mode "${LAMBDA_MODE}" \
  --lambda-grid ${LAMBDA_GRID} \
  --lambda-eval-step-size "${LAMBDA_EVAL_STEP_SIZE}" \
  --replay-backend "${REPLAY_BACKEND}" \
  --mean-flow-n-steps "${MEAN_FLOW_N_STEPS}" \
  --mean-flow-train-steps "${MEAN_FLOW_TRAIN_STEPS}" \
  --no-eval-fid \
  --no-compile
