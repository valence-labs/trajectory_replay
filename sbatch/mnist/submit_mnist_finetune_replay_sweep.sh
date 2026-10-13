#!/usr/bin/env bash
# Finetune sweep: generative replay fractions r50, r75, r90, and r99 (GR/GFR × cond/uncond).
# Results: results_dir/mnist/mnist_ft_*_d8_r{50,75,90,99}/
#
#   ./sbatch/mnist/submit_mnist_finetune_replay_sweep.sh
#   DRY_RUN=1 ./sbatch/mnist/submit_mnist_finetune_replay_sweep.sh
#
# Requires pretrain checkpoints:
#   results_dir/mnist/mnist_pretrain_d8/
#   results_dir/mnist/mnist_pretrain_uncond_d8/

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
export RESULTS_PREFIX="${RESULTS_PREFIX:-mnist}"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_mnist_finetune.sbatch"

submit_one() {
  local run_suffix="$1"
  local pretrain_run="$2"
  local strategy="$3"
  local unconditional="$4"
  local replay_fraction="$5"
  local job_name="$6"

  local run_name="${RESULTS_PREFIX}/mnist_ft_${run_suffix}_d${HELD_OUT_DIGIT}"
  local log_file="${SBATCH_ROOT}/logs/mnist_finetune_${run_suffix}_%j.log"

  local sbatch_cmd=(
    sbatch
    --export=ALL,RUN_NAME="${run_name}",PRETRAIN_RUN="${pretrain_run}",FINETUNE_STRATEGY="${strategy}",UNCONDITIONAL="${unconditional}",GENERATIVE_REPLAY_FRACTION="${replay_fraction}"
    --job-name="${job_name}"
    --output="${log_file}"
    "${SBATCH_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${sbatch_cmd[*]}"
  else
    "${sbatch_cmd[@]}"
    echo "Submitted: RUN_NAME=${run_name} PRETRAIN=${pretrain_run} FRACTION=${replay_fraction}"
  fi
}

for entry in r50:0.50 r75:0.75 r90:0.90 r99:0.99; do
  frac_label="${entry%%:*}"
  replay_fraction="${entry##*:}"
  submit_one \
    "generative_replay_cond_${frac_label}" \
    "${RESULTS_PREFIX}/mnist_pretrain_d8" \
    "generative_replay" \
    "0" \
    "${replay_fraction}" \
    "mnist-ft-gr-cond-${frac_label}"

  submit_one \
    "generative_field_replay_cond_${frac_label}" \
    "${RESULTS_PREFIX}/mnist_pretrain_d8" \
    "generative_field_replay" \
    "0" \
    "${replay_fraction}" \
    "mnist-ft-gfr-cond-${frac_label}"

  submit_one \
    "generative_replay_uncond_${frac_label}" \
    "${RESULTS_PREFIX}/mnist_pretrain_uncond_d8" \
    "generative_replay" \
    "1" \
    "${replay_fraction}" \
    "mnist-ft-gr-uncond-${frac_label}"

  submit_one \
    "generative_field_replay_uncond_${frac_label}" \
    "${RESULTS_PREFIX}/mnist_pretrain_uncond_d8" \
    "generative_field_replay" \
    "1" \
    "${replay_fraction}" \
    "mnist-ft-gfr-uncond-${frac_label}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Logs: ${SBATCH_ROOT}/logs/mnist_finetune_*_<jobid>.log"
  echo "Results under results_dir/${RESULTS_PREFIX}/mnist_ft_*_d${HELD_OUT_DIGIT}_r{50,75,90,99}/"
fi
