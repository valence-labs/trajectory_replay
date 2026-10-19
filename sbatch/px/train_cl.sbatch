#!/usr/bin/env bash
#SBATCH --job-name=px-cl
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=sbatch/logs/px_cl_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${STRATEGY:=trajectory}"
: "${SEED:=42}"
: "${REPLAY_BACKEND:=ode}"
: "${MEAN_FLOW_N_STEPS:=1}"
: "${MEAN_FLOW_BATCH_SIZE:=16}"
: "${LAMBDA_MODE:=equal}"
: "${LAMBDA_GRID:=0.25 0.5 0.75}"
: "${LAMBDA_SCHEDULE:=}"
: "${STEPS_PER_TASK:=25000}"
: "${BATCH_SIZE:=64}"
: "${MAX_TASKS:=3}"
: "${EVAL_FID:=1}"
: "${TASK0_CHECKPOINT:=}"
: "${TASK0_MEAN_FLOW:=}"
: "${EXPORT_TASK0_DIR:=}"
: "${OUTPUT_DIR:=results_dir/px/cl/${STRATEGY}_${REPLAY_BACKEND}_${LAMBDA_MODE}_s${SEED}}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

extra=()
[[ "${EVAL_FID}" == "1" ]] || extra+=(--no-eval-fid)
[[ -z "${TASK0_CHECKPOINT}" ]] || extra+=(--task0-checkpoint "${TASK0_CHECKPOINT}")
[[ -z "${TASK0_MEAN_FLOW}" ]] || extra+=(--task0-mean-flow "${TASK0_MEAN_FLOW}")
[[ -z "${EXPORT_TASK0_DIR}" ]] || extra+=(--export-task0-dir "${EXPORT_TASK0_DIR}")
[[ -z "${LAMBDA_SCHEDULE}" ]] || extra+=(--lambda-schedule ${LAMBDA_SCHEDULE})

uv run python experiments/px/train_cl.py \
  --launcher.cluster debug \
  --launcher.output_dir "${OUTPUT_DIR}" \
  --strategy "${STRATEGY}" \
  --model-name ADM \
  --seed "${SEED}" \
  --steps-per-task "${STEPS_PER_TASK}" \
  --batch-size "${BATCH_SIZE}" \
  --max-tasks "${MAX_TASKS}" \
  --replay-backend "${REPLAY_BACKEND}" \
  --mean-flow-n-steps "${MEAN_FLOW_N_STEPS}" \
  --mean-flow-batch-size "${MEAN_FLOW_BATCH_SIZE}" \
  --mean-flow-train-steps "${STEPS_PER_TASK}" \
  --lambda-mode "${LAMBDA_MODE}" \
  --lambda-grid ${LAMBDA_GRID} \
  --replay-step-size 0.05 \
  --lambda-eval-step-size 0.05 \
  "${extra[@]}"
