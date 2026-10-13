#!/usr/bin/env bash
# Top-panel triplicates: shared pretrain + seeded FT grid + seeded oracles.
#
# Pretrain is NOT re-run by default (frozen old model). Set RUN_PRETRAIN=1 to
# train/refresh pretrain_task0_seed0 first.
#
#   ./sbatch/sim_mixture/submit_replicates.sh
#   DRY_RUN=1 ./sbatch/sim_mixture/submit_replicates.sh
#   SEEDS="0 1 2" EPOCHS=200 ./sbatch/sim_mixture/submit_replicates.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
RUN_PRETRAIN="${RUN_PRETRAIN:-0}"
export SIM_ROOT
export DIM="${DIM:-2}"
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export DEVICE="${DEVICE:-cuda}"
export SEEDS="${SEEDS:-0 1 2}"
export STRATEGIES="${STRATEGIES:-naive,generative,trajectory}"
export PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${SIM_ROOT}/results_dir/sim_mixture/dim${DIM}/pretrain_task0_seed0/checkpoint.pt}"

echo "=== sim_mixture replicates (shared pretrain) ==="
echo "SEEDS=${SEEDS}  EPOCHS=${EPOCHS}  DIM=${DIM}  STRATEGIES=${STRATEGIES}"
echo "PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT}"
echo "RUN_PRETRAIN=${RUN_PRETRAIN}"

DEP=""
if [[ "${RUN_PRETRAIN}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    SEEDS="0" DRY_RUN=1 "${SCRIPT_DIR}/submit_pretrain.sh"
    DEP="DRY"
  else
    jid="$(SEEDS="0" "${SCRIPT_DIR}/submit_pretrain.sh" | tr -d '[:space:]')"
    if [[ ! "${jid}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: bad pretrain job id: '${jid}'" >&2
      exit 1
    fi
    DEP="afterok:${jid}"
    echo "Pretrain job ${jid} -> will gate FT"
  fi
elif [[ "${DRY_RUN}" != "1" && ! -f "${PRETRAIN_CHECKPOINT}" ]]; then
  echo "ERROR: missing ${PRETRAIN_CHECKPOINT} (set RUN_PRETRAIN=1 to train it)" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_RUN=1 "${SCRIPT_DIR}/submit_ft_lambda_grid.sh"
  ALL_LAMBDAS=1 DRY_RUN=1 "${SCRIPT_DIR}/submit_oracle_mixture.sh"
else
  if [[ -n "${DEP}" && "${DEP}" != "DRY" ]]; then
    DEPENDENCY="${DEP}" "${SCRIPT_DIR}/submit_ft_lambda_grid.sh"
  else
    "${SCRIPT_DIR}/submit_ft_lambda_grid.sh"
  fi
  ALL_LAMBDAS=1 "${SCRIPT_DIR}/submit_oracle_mixture.sh"
fi

echo "=== submit_replicates done ==="
echo "Results under results_dir/sim_mixture/dim${DIM}/"
