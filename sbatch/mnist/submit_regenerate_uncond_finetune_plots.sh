#!/usr/bin/env bash
# Regenerate KNN-stratified seen-digit plots for all unconditional finetune runs.
#
#   ./sbatch/mnist/submit_regenerate_uncond_finetune_plots.sh
#   DRY_RUN=1 ./sbatch/mnist/submit_regenerate_uncond_finetune_plots.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DEVICE="${DEVICE:-cuda}"
export RESULTS_PREFIX="${RESULTS_PREFIX:-mnist}"

SBATCH_SCRIPT="${SCRIPT_DIR}/regenerate_seen_digit_plots.sbatch"

submit_one() {
  local run_suffix="$1"
  local job_name="$2"
  local run_name="${RESULTS_PREFIX}/mnist_ft_${run_suffix}_d8"
  local log_file="${SBATCH_ROOT}/logs/mnist_seen_plots_${run_suffix}_%j.log"

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

submit_one "generative_replay_uncond" "mnist-plots-gr-uncond"
submit_one "generative_field_replay_uncond" "mnist-plots-gfr-uncond"
submit_one "generative_replay_uncond_r50" "mnist-plots-gr-uncond-r50"
submit_one "generative_field_replay_uncond_r50" "mnist-plots-gfr-uncond-r50"
submit_one "generative_replay_uncond_r75" "mnist-plots-gr-uncond-r75"
submit_one "generative_field_replay_uncond_r75" "mnist-plots-gfr-uncond-r75"
submit_one "generative_replay_uncond_r90" "mnist-plots-gr-uncond-r90"
submit_one "generative_field_replay_uncond_r90" "mnist-plots-gfr-uncond-r90"

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Logs: ${SBATCH_ROOT}/logs/mnist_seen_plots_*_<jobid>.log"
fi
