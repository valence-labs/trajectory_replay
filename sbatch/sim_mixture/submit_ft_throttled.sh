#!/usr/bin/env bash
# Submit FT grid with at most MAX_CONCURRENT jobs in the queue/running.
#
#   MAX_CONCURRENT=10 SEEDS="0 1 2" EPOCHS=200 ./sbatch/sim_mixture/submit_ft_throttled.sh

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
export LR="${LR:-1e-3}"
export DEVICE="${DEVICE:-cuda}"
export PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${SIM_ROOT}/results_dir/sim_mixture/dim${DIM}/pretrain_task0_seed0/checkpoint.pt}"

LAMBDAS=(${LAMBDAS:-0.1 0.3 0.5 0.7 0.9})
IFS=',' read -r -a STRATEGIES <<< "${STRATEGIES:-naive,generative,trajectory}"
SEEDS=(${SEEDS:-0 1 2})

if [[ ! -f "${PRETRAIN_CHECKPOINT}" ]]; then
  echo "ERROR: missing pretrain checkpoint: ${PRETRAIN_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${SBATCH_ROOT}/logs"

# Build (strategy, lambda, seed) work list. lambda empty for naive.
WORK=()
for seed in "${SEEDS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    strategy="$(echo "${strategy}" | tr -d '[:space:]')"
    if [[ "${strategy}" == "naive" ]]; then
      WORK+=("${strategy}|-|${seed}")
    else
      for lam in "${LAMBDAS[@]}"; do
        WORK+=("${strategy}|${lam}|${seed}")
      done
    fi
  done
done

echo "=== throttled FT submit ==="
echo "n_jobs=${#WORK[@]}  MAX_CONCURRENT=${MAX_CONCURRENT}  EPOCHS=${EPOCHS}"
echo "PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT}"

count_active() {
  # Count our simmix FT jobs (exclude oracle/pretrain).
  squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -cE '^simmix-(naive|generative|trajectory)' || true
}

submit_one() {
  local strategy="$1"
  local lam="$2"
  local seed="$3"
  export STRATEGY="${strategy}"
  export TRAIN_SEED="${seed}"
  export PRETRAIN_CHECKPOINT
  if [[ "${lam}" == "-" ]]; then
    export LAMBDA_MIX=0.5
    local tag="naive"
  else
    export LAMBDA_MIX="${lam}"
    local tag="${strategy}_lam${lam}"
  fi
  export RUN_NAME="ft_${tag}_e${EPOCHS}_seed${seed}"
  local log_file="${SBATCH_ROOT}/logs/sim_mixture_${RUN_NAME//./p}_%j.log"
  local jid
  jid="$(sbatch --export=ALL --parsable \
    --job-name="simmix-${tag//./p}-e${EPOCHS}-s${seed}" \
    --output="${log_file}" \
    "${SCRIPT_DIR}/train_finetune.sbatch")"
  echo "Submitted ${RUN_NAME} job=${jid}  (active~$(count_active))"
}

idx=0
n="${#WORK[@]}"
while (( idx < n )); do
  active="$(count_active)"
  active="${active:-0}"
  while (( active < MAX_CONCURRENT && idx < n )); do
    IFS='|' read -r strategy lam seed <<< "${WORK[$idx]}"
    submit_one "${strategy}" "${lam}" "${seed}"
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

echo "=== all ${n} FT jobs submitted ==="
echo "Waiting for queue to drain (optional follow)..."
while true; do
  active="$(count_active)"
  active="${active:-0}"
  if (( active == 0 )); then
    echo "All FT jobs finished."
    break
  fi
  echo "  still active=${active}; sleeping ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

echo "Results: results_dir/sim_mixture/dim${DIM}/finetune/"
done_n="$(find "results_dir/sim_mixture/dim${DIM}/finetune" -name metrics.json 2>/dev/null | wc -l || echo 0)"
echo "metrics.json count: ${done_n}/${n}"
