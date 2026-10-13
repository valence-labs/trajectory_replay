#!/bin/bash
#SBATCH --job-name=simcond-b-ft
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/sim_conditional_stage_b_ft_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${N_CONTEXTS:=6}"
: "${N_PERTURBATIONS:=4}"
: "${STRATEGY:=trajectory}"
: "${LAMBDA_MIX:=0.5}"
: "${EPOCHS:=2000}"
: "${BATCH_SIZE:=1024}"
: "${LR:=3e-4}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cuda}"
: "${N_GEN_EVAL:=500}"
: "${ODE_METHOD:=euler}"
: "${ODE_STEP_SIZE:=0.05}"
: "${TRAJ_KAPPA:=none}"
: "${PRETRAIN_CHECKPOINT:=}"
: "${RUN_NAME:=}"

extra=()
if [[ -n "${PRETRAIN_CHECKPOINT}" ]]; then
  extra+=(--pretrain-checkpoint "${PRETRAIN_CHECKPOINT}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  extra+=(--run-name "${RUN_NAME}")
fi
if [[ "${STRATEGY}" == "trajectory" ]]; then
  extra+=(--traj-kappa "${TRAJ_KAPPA}")
fi

echo "Job ${SLURM_JOB_ID:-local} Stage B FT at $(date -Is)"
echo "DIM=${DIM} C=${N_CONTEXTS} P=${N_PERTURBATIONS} STRATEGY=${STRATEGY} LAMBDA_MIX=${LAMBDA_MIX} EPOCHS=${EPOCHS}"

# Avoid leaking Cursor sandbox UV cache paths from the submitter env (--export=ALL).
unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

uv sync
uv run python experiments/sim_conditional/stage_b_shift/train_finetune.py \
  --dim "${DIM}" \
  --n-contexts "${N_CONTEXTS}" \
  --n-perturbations "${N_PERTURBATIONS}" \
  --strategy "${STRATEGY}" \
  --lambda-mix "${LAMBDA_MIX}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  --ode-method "${ODE_METHOD}" \
  --ode-step-size "${ODE_STEP_SIZE}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
