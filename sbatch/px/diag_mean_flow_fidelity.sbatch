#!/bin/bash
# Does the distilled task-0 MeanFlow track the ODE teacher on RxRx1?
#
#   sbatch sbatch/px/diag_mean_flow_fidelity.sbatch
#   SEEDS="42" sbatch sbatch/px/diag_mean_flow_fidelity.sbatch
#SBATCH --job-name=px-mf-fidelity
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/px_mf_fidelity_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

: "${SEEDS:=42 69 1714}"
: "${JUMPS:=1 4}"
: "${N:=256}"

# shellcheck disable=SC2086
"${PYTHON_BIN}" experiments/px/diag_mean_flow_fidelity.py \
  --seeds ${SEEDS} \
  --jumps ${JUMPS} \
  --n "${N}"

echo "Finished at $(date -Is)"
