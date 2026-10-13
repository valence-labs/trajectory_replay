#!/bin/bash
#SBATCH --job-name=simseq-cl
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=def
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch/logs/sim_sequential_%j.log

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p sbatch/logs

: "${STRATEGY:=trajectory}"
: "${LAMBDA_MODE:=equal}"
: "${LAMBDA_CONSTANT:=}"
: "${LAMBDA_GRID:=0.25 0.5 0.6666666667 0.75 0.8 0.9}"
: "${STEPS_PER_TASK:=20000}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${MAX_GRAD_NORM:=1.0}"
: "${SEED:=0}"
: "${DEVICE:=cuda}"
: "${ODE_STEP_SIZE:=0.05}"
: "${N_GEN_SELECT:=4000}"
: "${N_GEN_TEST:=10000}"
: "${RUNS_DIR:=}"
: "${RUN_NAME:=}"

args=(
  --strategy "${STRATEGY}"
  --lambda-mode "${LAMBDA_MODE}"
  --steps-per-task "${STEPS_PER_TASK}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --ode-step-size "${ODE_STEP_SIZE}"
  --n-gen-select "${N_GEN_SELECT}"
  --n-gen-test "${N_GEN_TEST}"
)

if [[ "${LAMBDA_MODE}" == "constant" ]]; then
  args+=(--lambda-constant "${LAMBDA_CONSTANT}")
elif [[ "${LAMBDA_MODE}" == "adaptive" ]]; then
  read -r -a lambda_grid <<< "${LAMBDA_GRID}"
  args+=(--lambda-grid "${lambda_grid[@]}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  args+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${RUNS_DIR}" ]]; then
  args+=(--runs-dir "${RUNS_DIR}")
fi

echo "Starting ${STRATEGY}/${LAMBDA_MODE}, seed=${SEED} at $(date -Is)"
"${SIM_ROOT}/.venv/bin/python" experiments/sim_sequential/train_cl.py "${args[@]}"
echo "Finished at $(date -Is)"
