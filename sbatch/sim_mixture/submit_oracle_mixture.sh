#!/usr/bin/env bash
# Submit oracle FM-on-mixture job(s).
#
#   LAMBDA_MIX=0.5 ./sbatch/sim_mixture/submit_oracle_mixture.sh
#   ALL_LAMBDAS=1 SEEDS="0 1 2" ./sbatch/sim_mixture/submit_oracle_mixture.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DIM="${DIM:-2}"
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export DEVICE="${DEVICE:-cuda}"

SEEDS=(${SEEDS:-0})
DEP_OPT=()
if [[ -n "${DEPENDENCY:-}" ]]; then
  DEP_OPT=(--dependency="${DEPENDENCY}")
fi

submit_one() {
  local lam="$1"
  local seed="$2"
  export LAMBDA_MIX="${lam}"
  export TRAIN_SEED="${seed}"
  local tag="lam${lam}"
  export RUN_NAME="oracle_mixture_${tag//./p}_seed${seed}"
  local log_file="${SBATCH_ROOT}/logs/sim_mixture_${RUN_NAME}_%j.log"
  local cmd=(
    sbatch
    --export=ALL
    --parsable
    --job-name="simmix-oracle-${tag//./p}-s${seed}"
    --output="${log_file}"
    "${DEP_OPT[@]}"
    "${SCRIPT_DIR}/train_oracle_mixture.sbatch"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${cmd[*]}  # λ=${lam} seed=${seed} RUN_NAME=${RUN_NAME}"
  else
    local jid
    jid="$("${cmd[@]}")"
    echo "Submitted oracle λ=${lam} seed=${seed} job=${jid} -> ${RUN_NAME}" >&2
  fi
}

LAM_LIST=()
if [[ "${ALL_LAMBDAS:-0}" == "1" ]]; then
  LAM_LIST=(0.1 0.3 0.5 0.7 0.9)
else
  LAM_LIST=("${LAMBDA_MIX:-0.5}")
fi

for seed in "${SEEDS[@]}"; do
  for lam in "${LAM_LIST[@]}"; do
    submit_one "${lam}" "${seed}"
  done
done
