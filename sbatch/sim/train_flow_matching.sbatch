#!/bin/bash
#SBATCH --job-name=flow-fm
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/%j.log
## GPU (example — adjust partition / gres for your site):
## #SBATCH --partition=def
## #SBATCH --gres=gpu:1

# Flow-matching training on the Gaussian SC simulator.
# Run from the simulation root (same layout as 2_pca_reconstruction_benchmark):
#   sbatch sbatch/sim/train_flow_matching.sbatch
# Override knobs, e.g.:
#   sbatch --export=ALL,EPOCHS=500,COUPLING=rotated sbatch/sim/train_flow_matching.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
SBATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

: "${RUN_NAME:=slurm_${SLURM_JOB_ID}}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_NAME="${RUN_NAME}_array${SLURM_ARRAY_TASK_ID}"
fi

: "${COUPLING:=additive}"
: "${N_CONTEXTS:=3}"
: "${N_PERTURBATIONS:=4}"
: "${DIM:=2}"
: "${SAMPLES_PER_COMBO:=200}"
: "${SIGMA:=0.18}"
: "${DATA_SEED:=0}"
: "${CONTEXT_SCALE:=1.2}"
: "${PERT_SCALE:=0.9}"
: "${INTERACTION_SCALE:=0.35}"

: "${COND_EMB_DIM:=32}"
: "${TIME_EMB_DIM:=64}"
: "${HIDDEN_DIM:=256}"
: "${DEPTH:=4}"

: "${EPOCHS:=200}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cpu}"

: "${ODE_METHOD:=dopri5}"
: "${ODE_RTOL:=1e-5}"
: "${ODE_ATOL:=1e-5}"
: "${ODE_STEP_SIZE:=}"

extra_ode=()
if [[ -n "${ODE_STEP_SIZE}" ]]; then
  extra_ode+=(--ode-step-size "${ODE_STEP_SIZE}")
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  RUN_NAME=${RUN_NAME}  DEVICE=${DEVICE}"

uv sync
uv run python experiments/sim/train_flow_matching.py \
  --coupling "${COUPLING}" \
  --n-contexts "${N_CONTEXTS}" \
  --n-perturbations "${N_PERTURBATIONS}" \
  --dim "${DIM}" \
  --samples-per-combo "${SAMPLES_PER_COMBO}" \
  --sigma "${SIGMA}" \
  --data-seed "${DATA_SEED}" \
  --context-scale "${CONTEXT_SCALE}" \
  --pert-scale "${PERT_SCALE}" \
  --interaction-scale "${INTERACTION_SCALE}" \
  --cond-emb-dim "${COND_EMB_DIM}" \
  --time-emb-dim "${TIME_EMB_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --eval-ode \
  --ode-method "${ODE_METHOD}" \
  --ode-rtol "${ODE_RTOL}" \
  --ode-atol "${ODE_ATOL}" \
  "${extra_ode[@]}"

echo "Finished at $(date -Is)"
echo "Results: $(pwd)/results_dir/${RUN_NAME}"
