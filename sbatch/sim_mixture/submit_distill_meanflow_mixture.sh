#!/usr/bin/env bash
# Submit MeanFlow-distilled-directly-on-mixture job(s) (no CL, no replay).
#
#   LAMBDA_MIX=0.1 ./sbatch/sim_mixture/submit_distill_meanflow_mixture.sh
#   ALL_LAMBDAS=1 ./sbatch/sim_mixture/submit_distill_meanflow_mixture.sh
#   DRY_RUN=1 ALL_LAMBDAS=1 ./sbatch/sim_mixture/submit_distill_meanflow_mixture.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DIM="${DIM:-2}"
export DISTILL_EPOCHS="${DISTILL_EPOCHS:-400}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-3}"
export TRAIN_SEED="${TRAIN_SEED:-0}"
export DIAG_FRACTION="${DIAG_FRACTION:-0.25}"
export CORNER_FRACTION="${CORNER_FRACTION:-0.0}"
export TIME_MU="${TIME_MU:--2.0}"
export TIME_SIGMA="${TIME_SIGMA:-2.0}"
export ADAPTIVE_P="${ADAPTIVE_P:-0.0}"
export DEVICE="${DEVICE:-cuda}"
export N_GEN_EVAL="${N_GEN_EVAL:-4000}"
export NFE_GRID="${NFE_GRID:-1,2,4,8}"
# zhuyu-cs/MeanFlow reference heuristics (EMA intentionally excluded).
export USE_ADAMW="${USE_ADAMW:-1}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

if [[ "${ALL_LAMBDAS:-0}" == "1" ]]; then
  LAMBDAS=(0.1 0.3 0.5 0.7 0.9)
else
  LAMBDAS=("${LAMBDA_MIX:-0.5}")
fi

submit_one() {
  local lam="$1"
  export LAMBDA_MIX="${lam}"
  local tag="lam${lam}"
  local heuristic_tag=""
  if [[ "${USE_ADAMW}" == "1" || "${MAX_GRAD_NORM}" != "0" ]]; then
    heuristic_tag="_adamw_b${ADAM_BETA2}_gc${MAX_GRAD_NORM}"
  fi
  local adapt_tag=""
  if [[ "${ADAPTIVE_P}" == "0" || "${ADAPTIVE_P}" == "0.0" ]]; then
    adapt_tag="_noadapt"
  else
    adapt_tag="_adaptp${ADAPTIVE_P}"
  fi
  local corner_tag=""
  if [[ "${CORNER_FRACTION}" != "0" && "${CORNER_FRACTION}" != "0.0" ]]; then
    corner_tag="_corner${CORNER_FRACTION}"
  fi
  local diag_tag=""
  if [[ "${DIAG_FRACTION}" != "0.25" ]]; then
    diag_tag="_diag${DIAG_FRACTION}"
  fi
  export RUN_NAME="distill_mixture_${tag//./p}_e${DISTILL_EPOCHS}_seed${TRAIN_SEED}${heuristic_tag//./p}${adapt_tag//./p}${corner_tag//./p}${diag_tag//./p}"
  local log_file="${SBATCH_ROOT}/logs/sim_mixture_${RUN_NAME}_%j.log"
  local cmd=(
    sbatch
    --export=ALL
    --parsable
    --job-name="simmix-distill-mix"
    --output="${log_file}"
    "${SCRIPT_DIR}/train_distill_meanflow_mixture.sbatch"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${cmd[*]}  # LAMBDA_MIX=${lam} RUN_NAME=${RUN_NAME}"
  else
    jid="$("${cmd[@]}")"
    echo "Submitted distill-on-mixture job=${jid} lambda=${lam} run=${RUN_NAME}" >&2
  fi
}

for lam in "${LAMBDAS[@]}"; do
  submit_one "${lam}"
done
