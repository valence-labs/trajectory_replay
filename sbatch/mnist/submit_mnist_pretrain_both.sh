#!/usr/bin/env bash
# Submit both MNIST pretrain jobs (conditional + unconditional).
#
#   ./sbatch/mnist/submit_mnist_pretrain_both.sh
#   DRY_RUN=1 ./sbatch/mnist/submit_mnist_pretrain_both.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DEVICE="${DEVICE:-cuda}"
export EPOCHS="${EPOCHS:-100}"
export HELD_OUT_DIGIT="${HELD_OUT_DIGIT:-8}"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_mnist_pretrain.sbatch"

submit_one() {
  local run_name="$1"
  local unconditional="$2"
  local job_name="$3"
  local log_file="${SBATCH_ROOT}/logs/mnist_pretrain_${run_name}_%j.log"

  local sbatch_cmd=(
    sbatch
    --export=ALL,RUN_NAME="${run_name}",UNCONDITIONAL="${unconditional}"
    --job-name="${job_name}"
    --output="${log_file}"
    "${SBATCH_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${sbatch_cmd[*]}"
  else
    "${sbatch_cmd[@]}"
    echo "Submitted: RUN_NAME=${run_name} UNCONDITIONAL=${unconditional}"
  fi
}

submit_one "mnist_pretrain_d8" "0" "mnist-pt-cond"
submit_one "mnist_pretrain_uncond_d8" "1" "mnist-pt-uncond"

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Logs: ${SBATCH_ROOT}/logs/mnist_pretrain_*_<jobid>.log"
  echo "Results: results_dir/mnist_pretrain_d8/ and results_dir/mnist_pretrain_uncond_d8/"
fi
