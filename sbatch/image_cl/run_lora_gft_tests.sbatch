#!/bin/bash
# CPU-only tests for the image-CL LoRA and gradual-finetuning strategies, plus a
# short GPU smoke run of both through train_cl.py. Never run these on a login node.
#
# Usage:
#   sbatch sbatch/image_cl/run_lora_gft_tests.sbatch
#   SMOKE=0 sbatch sbatch/image_cl/run_lora_gft_tests.sbatch   # unit tests only
#SBATCH --job-name=image-cl-lora-gft-tests
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/image_cl_lora_gft_tests_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
# Only script-runnable tests belong here: files without a __main__ block exit 0
# without running anything, which would report a vacuous PASS.
: "${TESTS:=tests/test_image_cl_lora_gft.py}"
: "${SMOKE:=1}"

export PYTHONPATH="src:.:${PYTHONPATH:-}"

status=0
for test_file in ${TESTS}; do
  echo "=== ${test_file} ===" >&2
  if CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${test_file}"; then
    echo "PASS ${test_file}" >&2
  else
    echo "FAIL ${test_file}" >&2
    status=1
  fi
done

if [[ "${SMOKE}" == "1" ]]; then
  for strategy in lora gft; do
    echo "=== smoke train_cl.py --strategy ${strategy} ===" >&2
    if "${PYTHON_BIN}" experiments/image_cl/train_cl.py \
      --dataset split_fmnist \
      --strategy "${strategy}" \
      --seed 42 \
      --smoke \
      --no-resume \
      --no-kld \
      --run-name "image_cl/fmnist/_smoke_${strategy}_s42"; then
      echo "PASS smoke ${strategy}" >&2
    else
      echo "FAIL smoke ${strategy}" >&2
      status=1
    fi
  done
fi

exit "${status}"
