#!/bin/bash
# CPU-only test suite for the Tx CL pipeline. Never run these on a login node;
# the trainer smoke tests spawn full train_cl.py subprocesses.
#
# Usage:
#   sbatch sbatch/tx/run_cl_tests.sbatch
#   TESTS="tests/test_tx_cl_stage_c.py" sbatch sbatch/tx/run_cl_tests.sbatch
#SBATCH --job-name=tx-cl-tests
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/tx_cl_tests_%j.log
#SBATCH --partition=def

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
: "${TESTS:=tests/test_tx_cl_stage_c.py tests/test_tx_cl_tasks.py tests/test_tx_cl_trainer_smoke.py}"

export PYTHONPATH="src:.:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

status=0
for test_file in ${TESTS}; do
  echo "=== ${test_file} ===" >&2
  if "${PYTHON_BIN}" "${test_file}"; then
    echo "PASS ${test_file}" >&2
  else
    echo "FAIL ${test_file}" >&2
    status=1
  fi
done
exit "${status}"
