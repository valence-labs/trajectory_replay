#!/bin/bash
#SBATCH --job-name=flow-holdout-cmp
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/%j.log
# #SBATCH --partition=def
# #SBATCH --gres=gpu:1

# Held-out (c,p) combo: train noise-to-data + data-to-data (combo excluded), compare on held-out.
#   sbatch sbatch/sim/train_combo_holdout_compare.sbatch
#
# Override held-out combo (default c=1, p=3; control p=0 for data-to-data):
#   sbatch --export=ALL,HELD_OUT_CONTEXT=2,HELD_OUT_PERT=3 sbatch/sim/train_combo_holdout_compare.sbatch
#
# Data-to-data without control→control identity training (compare in one job):
#   sbatch --export=ALL,TRAIN_BOTH_D2D_VARIANTS=1 sbatch/sim/train_combo_holdout_compare.sbatch
# Or only the no-identity variant:
#   sbatch --export=ALL,D2D_EXCLUDE_IDENTITY_TARGETS=1 sbatch/sim/train_combo_holdout_compare.sbatch
#
# Save model checkpoints (default off):
#   sbatch --export=ALL,SAVE_CHECKPOINTS=1 sbatch/sim/train_combo_holdout_compare.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
SBATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

: "${RUN_NAME:=slurm_holdout_cmp_${SLURM_JOB_ID}}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_NAME="${RUN_NAME}_array${SLURM_ARRAY_TASK_ID}"
fi

: "${COUPLING:=rotated}"
: "${N_CONTEXTS:=10}"
: "${N_PERTURBATIONS:=4}"
: "${DIM:=2}"
: "${SAMPLES_PER_COMBO:=200}"
: "${SIGMA:=0.18}"
: "${DATA_SEED:=0}"

: "${HELD_OUT_CONTEXT:=1}"
: "${HELD_OUT_PERT:=3}"
: "${CONTROL_PERT:=0}"

: "${CONTEXT_SCALE:=1.2}"
: "${PERT_SCALE:=0.9}"
: "${INTERACTION_SCALE:=0.80}"

: "${COND_EMB_DIM:=32}"
: "${TIME_EMB_DIM:=64}"
: "${HIDDEN_DIM:=256}"
: "${DEPTH:=4}"

: "${EPOCHS:=2000}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cpu}"

: "${ODE_METHOD:=dopri5}"
: "${ODE_RTOL:=1e-5}"
: "${ODE_ATOL:=1e-5}"
: "${ODE_STEP_SIZE:=}"
: "${NOISE_SCALE:=1.0}"

: "${D2D_EXCLUDE_IDENTITY_TARGETS:=0}"
: "${TRAIN_BOTH_D2D_VARIANTS:=0}"
: "${SAVE_CHECKPOINTS:=0}"
: "${SWEEP_REGIME:=}"
: "${CONTROL_SUBSAMPLE_FRAC:=}"

extra_d2d=()
if [[ "${TRAIN_BOTH_D2D_VARIANTS}" == "1" ]]; then
  extra_d2d+=(--train-both-d2d-variants)
elif [[ "${D2D_EXCLUDE_IDENTITY_TARGETS}" == "1" ]]; then
  extra_d2d+=(--d2d-exclude-identity-targets)
fi

extra_ode=()
if [[ -n "${ODE_STEP_SIZE}" ]]; then
  extra_ode+=(--ode-step-size "${ODE_STEP_SIZE}")
fi

extra_ckpt=()
if [[ "${SAVE_CHECKPOINTS}" == "1" ]]; then
  extra_ckpt+=(--save-checkpoints)
fi

extra_sweep=()
if [[ -n "${SWEEP_REGIME}" ]]; then
  extra_sweep+=(--sweep-regime "${SWEEP_REGIME}")
fi

extra_ctrl=()
if [[ -n "${CONTROL_SUBSAMPLE_FRAC}" ]]; then
  extra_ctrl+=(--control-subsample-frac "${CONTROL_SUBSAMPLE_FRAC}")
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  RUN_NAME=${RUN_NAME}"
echo "Held-out (c,p)=(${HELD_OUT_CONTEXT},${HELD_OUT_PERT})  CONTROL_PERT=${CONTROL_PERT}"
echo "D2D_EXCLUDE_IDENTITY_TARGETS=${D2D_EXCLUDE_IDENTITY_TARGETS}  TRAIN_BOTH_D2D_VARIANTS=${TRAIN_BOTH_D2D_VARIANTS}"
if [[ -n "${CONTROL_SUBSAMPLE_FRAC}" ]]; then
  echo "CONTROL_SUBSAMPLE_FRAC=${CONTROL_SUBSAMPLE_FRAC} (drop control rows; noise + d2d)"
fi

uv sync
uv run python experiments/sim/train_combo_holdout_compare.py \
  --coupling "${COUPLING}" \
  --n-contexts "${N_CONTEXTS}" \
  --n-perturbations "${N_PERTURBATIONS}" \
  --dim "${DIM}" \
  --samples-per-combo "${SAMPLES_PER_COMBO}" \
  --sigma "${SIGMA}" \
  --data-seed "${DATA_SEED}" \
  --held-out-context "${HELD_OUT_CONTEXT}" \
  --held-out-pert "${HELD_OUT_PERT}" \
  --control-pert "${CONTROL_PERT}" \
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
  --noise-scale "${NOISE_SCALE}" \
  --run-name "${RUN_NAME}" \
  --ode-method "${ODE_METHOD}" \
  --ode-rtol "${ODE_RTOL}" \
  --ode-atol "${ODE_ATOL}" \
  "${extra_d2d[@]}" \
  "${extra_ckpt[@]}" \
  "${extra_sweep[@]}" \
  "${extra_ctrl[@]}" \
  "${extra_ode[@]}"

echo "Finished at $(date -Is)"
echo "Results: $(pwd)/results_dir/${RUN_NAME}"
echo "  heldout_compare.png — true vs data-to-data vs noise-to-data on held-out (c,p)"
echo "  metrics.json — energy distance + training history"
