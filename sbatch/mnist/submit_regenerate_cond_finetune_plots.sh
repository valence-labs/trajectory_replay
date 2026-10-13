#!/usr/bin/env bash
# Regenerate seen-digit plots for both conditional finetune runs (GPU jobs).
#
#   ./sbatch/mnist/submit_regenerate_cond_finetune_plots.sh
#   DRY_RUN=1 ./sbatch/mnist/submit_regenerate_cond_finetune_plots.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DEVICE="${DEVICE:-cuda}"

SBATCH_SCRIPT="${SCRIPT_DIR}/regenerate_seen_digit_plots.sbatch"

submit_one() {
  local run_name="$1"
  local job_name="$2"
  local log_file="${SBATCH_ROOT}/logs/mnist_seen_plots_${run_name}_%j.log"

  local sbatch_cmd=(
    sbatch
    --export=ALL,RUN_NAME="${run_name}"
    --job-name="${job_name}"
    --output="${log_file}"
    "${SBATCH_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${sbatch_cmd[*]}"
  else
    "${sbatch_cmd[@]}"
    echo "Submitted: RUN_NAME=${run_name}"
  fi
}

submit_one "mnist_ft_generative_replay_cond_d8" "mnist-plots-gr-cond"
submit_one "mnist_ft_generative_field_replay_cond_d8" "mnist-plots-gfr-cond"

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Logs: ${SBATCH_ROOT}/logs/mnist_seen_plots_*_<jobid>.log"
fi
