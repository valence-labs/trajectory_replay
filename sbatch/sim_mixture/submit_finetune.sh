#!/usr/bin/env bash
# Submit one sim_mixture finetune job.
#
#   STRATEGY=trajectory LAMBDA_MIX=0.5 ./sbatch/sim_mixture/submit_finetune.sh
#   STRATEGY=naive ./sbatch/sim_mixture/submit_finetune.sh
#   DRY_RUN=1 STRATEGY=generative LAMBDA_MIX=0.7 ./sbatch/sim_mixture/submit_finetune.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"

export SIM_ROOT
export DIM="${DIM:-2}"
export STRATEGY="${STRATEGY:?Set STRATEGY=naive|generative|trajectory|real}"
export LAMBDA_MIX="${LAMBDA_MIX:-0.5}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-3}"
export DEVICE="${DEVICE:-cuda}"
export REPLAY_BACKEND="${REPLAY_BACKEND:-ode}"
export MEAN_FLOW_STEPS="${MEAN_FLOW_STEPS:-1}"
export DIAG_FRACTION="${DIAG_FRACTION:-0.25}"
export TIME_MU="${TIME_MU:-2.0}"
export TIME_SIGMA="${TIME_SIGMA:-2.0}"
export ADAPTIVE_P="${ADAPTIVE_P:-0.75}"
export MF_MAX_GRAD_NORM="${MF_MAX_GRAD_NORM:-1.0}"
export MF_USE_ADAMW="${MF_USE_ADAMW:-1}"
export MF_ADAM_BETA2="${MF_ADAM_BETA2:-0.95}"
export MF_WEIGHT_DECAY="${MF_WEIGHT_DECAY:-0.0}"
[[ -n "${MF_LR:-}" ]] && export MF_LR
[[ -n "${MEAN_FLOW_CHECKPOINT:-}" ]] && export MEAN_FLOW_CHECKPOINT

tag="${STRATEGY}"
if [[ "${STRATEGY}" != "naive" ]]; then
  tag="${STRATEGY}_lam${LAMBDA_MIX}"
fi
if [[ "${REPLAY_BACKEND}" == "mean_flow" ]]; then
  tag="${tag}_mf_cifar"
fi
LOG_FILE="${SBATCH_ROOT}/logs/sim_mixture_ft_${tag//./p}_%j.log"

sbatch_cmd=(
  sbatch
  --export=ALL
  --job-name="simmix-ft-${STRATEGY}"
  --output="${LOG_FILE}"
  "${SCRIPT_DIR}/train_finetune.sbatch"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "${sbatch_cmd[*]}"
else
  "${sbatch_cmd[@]}"
  echo "Submitted finetune: STRATEGY=${STRATEGY} LAMBDA_MIX=${LAMBDA_MIX}"
fi
