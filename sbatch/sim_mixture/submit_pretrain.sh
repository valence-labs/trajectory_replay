#!/usr/bin/env bash
# Submit sim_mixture unconditional pretrain (task 0).
#
#   ./sbatch/sim_mixture/submit_pretrain.sh
#   SEEDS="0 1 2" ./sbatch/sim_mixture/submit_pretrain.sh
#   DRY_RUN=1 ./sbatch/sim_mixture/submit_pretrain.sh

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
export LR="${LR:-1e-3}"
export DEVICE="${DEVICE:-cuda}"
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export DEPTH="${DEPTH:-4}"
export TIME_EMB_DIM="${TIME_EMB_DIM:-64}"
export N_GEN_EVAL="${N_GEN_EVAL:-4000}"

SEEDS=(${SEEDS:-0})

submit_one() {
  local seed="$1"
  export TRAIN_SEED="${seed}"
  export RUN_NAME="${RUN_NAME_PREFIX:-pretrain_task0}_seed${seed}"
  local log_file="${SBATCH_ROOT}/logs/sim_mixture_${RUN_NAME}_%j.log"
  local cmd=(
    sbatch
    --export=ALL
    --parsable
    --job-name="simmix-pretrain-s${seed}"
    --output="${log_file}"
    "${SCRIPT_DIR}/train_pretrain.sbatch"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${cmd[*]}  # RUN_NAME=${RUN_NAME} TRAIN_SEED=${seed}"
  else
    local jid
    jid="$("${cmd[@]}")"
    echo "Submitted pretrain seed=${seed} job=${jid} -> ${RUN_NAME}" >&2
    echo "${jid}"
  fi
}

for seed in "${SEEDS[@]}"; do
  submit_one "${seed}"
done
