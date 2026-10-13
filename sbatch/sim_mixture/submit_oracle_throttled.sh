#!/usr/bin/env bash
# Submit oracle mixture grid with at most MAX_CONCURRENT jobs.
#
#   MAX_CONCURRENT=10 SEEDS="0 1 2" EPOCHS=200 ./sbatch/sim_mixture/submit_oracle_throttled.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
POLL_SEC="${POLL_SEC:-30}"
export SIM_ROOT
export DIM="${DIM:-2}"
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export DEVICE="${DEVICE:-cuda}"

LAMBDAS=(${LAMBDAS:-0.1 0.3 0.5 0.7 0.9})
SEEDS=(${SEEDS:-0 1 2})

mkdir -p "${SBATCH_ROOT}/logs"

WORK=()
for seed in "${SEEDS[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    WORK+=("${lam}|${seed}")
  done
done

echo "=== throttled oracle submit ==="
echo "n_jobs=${#WORK[@]}  MAX_CONCURRENT=${MAX_CONCURRENT}  EPOCHS=${EPOCHS}"

count_active() {
  squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -cE '^simmix-oracle-' || true
}

submit_one() {
  local lam="$1"
  local seed="$2"
  export LAMBDA_MIX="${lam}"
  export TRAIN_SEED="${seed}"
  export RUN_NAME="oracle_mixture_lam${lam}_e${EPOCHS}_seed${seed}"
  export RUN_NAME="${RUN_NAME//./p}"
  local log_file="${SBATCH_ROOT}/logs/sim_mixture_${RUN_NAME}_%j.log"
  local jid
  jid="$(sbatch --export=ALL --parsable \
    --job-name="simmix-oracle-lam${lam//./p}-e${EPOCHS}-s${seed}" \
    --output="${log_file}" \
    "${SCRIPT_DIR}/train_oracle_mixture.sbatch")"
  echo "Submitted ${RUN_NAME} job=${jid}  (active~$(count_active))"
}

idx=0
n="${#WORK[@]}"
while (( idx < n )); do
  active="$(count_active)"
  active="${active:-0}"
  while (( active < MAX_CONCURRENT && idx < n )); do
    IFS='|' read -r lam seed <<< "${WORK[$idx]}"
    submit_one "${lam}" "${seed}"
    idx=$((idx + 1))
    active="$(count_active)"
    active="${active:-0}"
  done
  if (( idx >= n )); then
    break
  fi
  echo "At capacity (${active}/${MAX_CONCURRENT}); sleeping ${POLL_SEC}s  (submitted ${idx}/${n})"
  sleep "${POLL_SEC}"
done

echo "=== all ${n} oracle jobs submitted ==="
while true; do
  active="$(count_active)"
  active="${active:-0}"
  if (( active == 0 )); then
    echo "All oracle jobs finished."
    break
  fi
  echo "  still active=${active}; sleeping ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

done_n="$(find "results_dir/sim_mixture/dim${DIM}/oracle_mixture" -name metrics.json 2>/dev/null | wc -l || echo 0)"
echo "metrics.json count: ${done_n}/${n}"
