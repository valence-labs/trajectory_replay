#!/usr/bin/env bash
#SBATCH --job-name=px-masip
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=sbatch/logs/px_masip_%j.log

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${STRATEGY:=generative_distillation}"
: "${SEED:=42}"
: "${STEPS_PER_TASK:=25000}"
: "${BATCH_SIZE:=32}"
: "${MAX_TASKS:=3}"
: "${EVAL_FID:=1}"
: "${TASK0_CHECKPOINT:=}"
: "${EXPORT_TASK0_DIR:=}"
: "${OUTPUT_DIR:=results_dir/px/masip/${STRATEGY}_s${SEED}}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

extra=()
[[ "${EVAL_FID}" == "1" ]] || extra+=(--no-eval-fid)
[[ -z "${TASK0_CHECKPOINT}" ]] || extra+=(--task0-checkpoint "${TASK0_CHECKPOINT}")
[[ -z "${EXPORT_TASK0_DIR}" ]] || extra+=(--export-task0-dir "${EXPORT_TASK0_DIR}")

uv run python experiments/px/train_masip_cl.py \
  --launcher.cluster debug \
  --launcher.output_dir "${OUTPUT_DIR}" \
  --strategy "${STRATEGY}" \
  --seed "${SEED}" \
  --steps-per-task "${STEPS_PER_TASK}" \
  --batch-size "${BATCH_SIZE}" \
  --max-tasks "${MAX_TASKS}" \
  --teacher-steps 20 \
  --eval-steps 50 \
  "${extra[@]}"
