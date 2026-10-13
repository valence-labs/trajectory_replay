#!/usr/bin/env bash
# Submit all four MNIST finetune jobs (2 replay strategies × conditional/unconditional).
#
#   ./sbatch/mnist/submit_mnist_finetune_all.sh
#   DRY_RUN=1 ./sbatch/mnist/submit_mnist_finetune_all.sh
#
# Requires pretrain checkpoints:
#   results_dir/mnist_pretrain_d8/
#   results_dir/mnist_pretrain_uncond_d8/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DEVICE="${DEVICE:-cuda}"
export HELD_OUT_DIGIT="${HELD_OUT_DIGIT:-8}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
export GENERATIVE_REPLAY_FRACTION="${GENERATIVE_REPLAY_FRACTION:-0.25}"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_mnist_finetune.sbatch"

submit_one() {
  local run_name="$1"
  local pretrain_run="$2"
  local strategy="$3"
  local unconditional="$4"
  local job_name="$5"
  local log_file="${SBATCH_ROOT}/logs/mnist_finetune_${run_name}_%j.log"

  local sbatch_cmd=(
    sbatch
    --export=ALL,RUN_NAME="${run_name}",PRETRAIN_RUN="${pretrain_run}",FINETUNE_STRATEGY="${strategy}",UNCONDITIONAL="${unconditional}"
    --job-name="${job_name}"
    --output="${log_file}"
    "${SBATCH_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${sbatch_cmd[*]}"
  else
    "${sbatch_cmd[@]}"
    echo "Submitted: RUN_NAME=${run_name} PRETRAIN_RUN=${pretrain_run} STRATEGY=${strategy} UNCONDITIONAL=${unconditional}"
  fi
}

submit_one \
  "mnist_ft_generative_replay_cond_d8" \
  "mnist_pretrain_d8" \
  "generative_replay" \
  "0" \
  "mnist-ft-gr-cond"

submit_one \
  "mnist_ft_generative_field_replay_cond_d8" \
  "mnist_pretrain_d8" \
  "generative_field_replay" \
  "0" \
  "mnist-ft-gfr-cond"

submit_one \
  "mnist_ft_generative_replay_uncond_d8" \
  "mnist_pretrain_uncond_d8" \
  "generative_replay" \
  "1" \
  "mnist-ft-gr-uncond"

submit_one \
  "mnist_ft_generative_field_replay_uncond_d8" \
  "mnist_pretrain_uncond_d8" \
  "generative_field_replay" \
  "1" \
  "mnist-ft-gfr-uncond"

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Logs: ${SBATCH_ROOT}/logs/mnist_finetune_*_<jobid>.log"
  echo "Results under results_dir/mnist_ft_*_d${HELD_OUT_DIGIT}/"
fi
