#!/bin/bash
#SBATCH --job-name=flow-forget
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/%j.log

# Hold-out pretrain + configurable finetune.
#   sbatch sbatch/sim/train_perturbation_forgetting.sbatch
#
# HOLDOUT_MODE=perturbation  — hold out one p in all contexts (default)
# HOLDOUT_MODE=context       — hold out one c in all perturbations (set HELD_OUT_CONTEXT)
# HOLDOUT_MODE=combo         — hold out one (c,p); p seen in other contexts (set HELD_OUT_CONTEXT)
#
# Finetune flags (0/false = off, 1/true = on):
#   FINETUNE_TRAIN_VF              — unfreeze MLP
#   FINETUNE_TRAIN_OLD_EMBEDDINGS  — unfreeze all context + perturbation rows
#   (default: held-out pert row, context row, or c+p rows per holdout mode)
#   FINETUNE_DATA=held_out|all|replay
#   REPLAY_FRACTION=0.25                   — replay mode: fraction of each batch from buffer
#   REPLAY_MAX_SAMPLES=600                 — optional cap (stratified per (c,p); empty = full pool)
#   REPLAY_SEED=                         — optional; default train_seed+17
#   SAVE_CHECKPOINT=1                      — write checkpoint_finetuned.pt (default: off)
#
# Examples:
#   sbatch --export=ALL,FINETUNE_TRAIN_VF=1 sbatch/sim/train_perturbation_forgetting.sbatch
#   sbatch --export=ALL,FINETUNE_TRAIN_OLD_EMBEDDINGS=1,FINETUNE_DATA=all sbatch/sim/train_perturbation_forgetting.sbatch
#   sbatch --export=ALL,FINETUNE_DATA=replay,REPLAY_FRACTION=0.25,REPLAY_MAX_SAMPLES=600 sbatch/sim/train_perturbation_forgetting.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
SBATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

: "${RUN_NAME:=slurm_forget_${SLURM_JOB_ID}}"
: "${HOLDOUT_MODE:=context}"
: "${HELD_OUT_PERT:=3}"
: "${HELD_OUT_CONTEXT:=2}"

: "${COUPLING:=additive}"
: "${N_CONTEXTS:=3}"
: "${N_PERTURBATIONS:=4}"
: "${DIM:=2}"
: "${SAMPLES_PER_COMBO:=200}"
: "${SIGMA:=0.25}"
: "${DATA_SEED:=0}"
: "${CONTEXT_SCALE:=4}"  # hard 5 easy 4
: "${PERT_SCALE:=0.5}"  # hard 1 easy 0.5
: "${INTERACTION_SCALE:=1}"

: "${COND_EMB_DIM:=32}"
: "${TIME_EMB_DIM:=64}"
: "${HIDDEN_DIM:=256}"
: "${DEPTH:=4}"

: "${PRETRAIN_EPOCHS:=200}"
: "${FINETUNE_EPOCHS:=200}"
: "${BATCH_SIZE:=256}"
: "${PRETRAIN_LR:=1e-3}"
: "${FINETUNE_LR:=1e-3}"
: "${FINETUNE_TRAIN_VF:=0}"
: "${FINETUNE_TRAIN_OLD_EMBEDDINGS:=0}"
: "${FINETUNE_DATA:=held_out}"
: "${REPLAY_FRACTION:=0.25}"
: "${REPLAY_MAX_SAMPLES:=}"
: "${REPLAY_SEED:=}"
: "${SAVE_CHECKPOINT:=0}"
: "${NEW_EMB_INIT_STD:=0.02}"
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

finetune_extra=()
if [[ "${FINETUNE_TRAIN_VF}" == "1" || "${FINETUNE_TRAIN_VF}" == "true" || "${FINETUNE_TRAIN_VF}" == "yes" ]]; then
  finetune_extra+=(--finetune-train-vf)
fi
if [[ "${FINETUNE_TRAIN_OLD_EMBEDDINGS}" == "1" || "${FINETUNE_TRAIN_OLD_EMBEDDINGS}" == "true" || "${FINETUNE_TRAIN_OLD_EMBEDDINGS}" == "yes" ]]; then
  finetune_extra+=(--finetune-train-old-embeddings)
fi
if [[ "${SAVE_CHECKPOINT}" == "1" || "${SAVE_CHECKPOINT}" == "true" || "${SAVE_CHECKPOINT}" == "yes" ]]; then
  finetune_extra+=(--save-checkpoint)
fi
if [[ "${FINETUNE_DATA}" == "replay" ]]; then
  finetune_extra+=(--replay-fraction "${REPLAY_FRACTION}")
  if [[ -n "${REPLAY_MAX_SAMPLES}" ]]; then
    finetune_extra+=(--replay-max-samples "${REPLAY_MAX_SAMPLES}")
  fi
  if [[ -n "${REPLAY_SEED}" ]]; then
    finetune_extra+=(--replay-seed "${REPLAY_SEED}")
  fi
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  RUN_NAME=${RUN_NAME}  HOLDOUT_MODE=${HOLDOUT_MODE}"
echo "  HELD_OUT_PERT=${HELD_OUT_PERT}  HELD_OUT_CONTEXT=${HELD_OUT_CONTEXT}"
echo "  FINETUNE_TRAIN_VF=${FINETUNE_TRAIN_VF}  FINETUNE_TRAIN_OLD_EMBEDDINGS=${FINETUNE_TRAIN_OLD_EMBEDDINGS}  FINETUNE_DATA=${FINETUNE_DATA}"
if [[ "${FINETUNE_DATA}" == "replay" ]]; then
  echo "  REPLAY_FRACTION=${REPLAY_FRACTION}  REPLAY_MAX_SAMPLES=${REPLAY_MAX_SAMPLES:-full pool}"
fi

uv sync
uv run python experiments/sim/train_perturbation_forgetting.py \
  --holdout-mode "${HOLDOUT_MODE}" \
  --held-out-pert "${HELD_OUT_PERT}" \
  --held-out-context "${HELD_OUT_CONTEXT}" \
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
  --pretrain-epochs "${PRETRAIN_EPOCHS}" \
  --finetune-epochs "${FINETUNE_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --pretrain-lr "${PRETRAIN_LR}" \
  --finetune-lr "${FINETUNE_LR}" \
  --finetune-data "${FINETUNE_DATA}" \
  --new-emb-init-std "${NEW_EMB_INIT_STD}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --ode-method "${ODE_METHOD}" \
  --ode-rtol "${ODE_RTOL}" \
  --ode-atol "${ODE_ATOL}" \
  "${finetune_extra[@]}" \
  "${extra_ode[@]}"

echo "Finished at $(date -Is)"
echo "Results: ./results_dir/${RUN_NAME}"
