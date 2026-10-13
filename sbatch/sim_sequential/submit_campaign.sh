#!/usr/bin/env bash
# Submit the three-seed sequential Gaussian-ring campaign.
#
# DRY_RUN=1 ./sbatch/sim_sequential/submit_campaign.sh
# ./sbatch/sim_sequential/submit_campaign.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
INCLUDE_NAIVE="${INCLUDE_NAIVE:-1}"
SEEDS=(${SEEDS:-0 1 2})
CONSTANT_LAMBDAS=(${CONSTANT_LAMBDAS:-0.25 0.5 0.75 0.9})
STRATEGIES=(${STRATEGIES:-generative trajectory})
export SIM_ROOT
export STEPS_PER_TASK="${STEPS_PER_TASK:-20000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-3}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export DEVICE="${DEVICE:-cuda}"
export ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.05}"
export N_GEN_SELECT="${N_GEN_SELECT:-4000}"
export N_GEN_TEST="${N_GEN_TEST:-10000}"
export LAMBDA_GRID="${LAMBDA_GRID:-0.25 0.5 0.6666666667 0.75 0.8 0.9}"
export RUNS_DIR="${RUNS_DIR:-}"

submit_one() {
  local strategy="$1"
  local mode="$2"
  local seed="$3"
  local lam="${4:-}"
  local tag="${strategy}_${mode}"
  export STRATEGY="${strategy}"
  export LAMBDA_MODE="${mode}"
  export SEED="${seed}"
  export LAMBDA_CONSTANT="${lam}"
  if [[ "${mode}" == "constant" ]]; then
    tag="${tag}_lam${lam//./p}"
  fi
  export RUN_NAME="${tag}_s${seed}"

  local cmd=(
    sbatch
    --parsable
    --export=ALL
    --job-name="simseq-${tag//./p}-s${seed}"
    --output="${SBATCH_ROOT}/logs/simseq_${tag//./p}_s${seed}_%j.log"
    "${SCRIPT_DIR}/train_cl.sbatch"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${cmd[*]}"
  else
    local job_id
    job_id="$("${cmd[@]}")"
    echo "Submitted ${RUN_NAME}: ${job_id}"
  fi
}

for seed in "${SEEDS[@]}"; do
  if [[ "${INCLUDE_NAIVE}" == "1" ]]; then
    submit_one naive none "${seed}"
  fi
  for strategy in "${STRATEGIES[@]}"; do
    for lam in "${CONSTANT_LAMBDAS[@]}"; do
      submit_one "${strategy}" constant "${seed}" "${lam}"
    done
    submit_one "${strategy}" equal "${seed}"
    submit_one "${strategy}" adaptive "${seed}"
  done
done
