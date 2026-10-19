#!/bin/bash
#SBATCH --job-name=tx-heldout-resid
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tx_heldout_residual_%j.log
#SBATCH --partition=YOUR_CPU_PARTITION

# Cross-line residual concordance and T0→HepG2 residual oracle (data only).
#
#   sbatch sbatch/tx/diag_heldout_residual_transfer.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/replogle

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
: "${H5AD_PATH:=${REPO_ROOT}/data_dir/replogle_with_splits.h5ad}"
: "${OUT_JSON:=${REPO_ROOT}/results_dir/tx_results/replogle/diag_heldout_residual_transfer.json}"

echo "H5AD_PATH=${H5AD_PATH}"
echo "OUT_JSON=${OUT_JSON}"

"${PYTHON_BIN}" experiments/tx/diag_heldout_residual_transfer.py \
  --h5ad-path "${H5AD_PATH}" \
  --out-json "${OUT_JSON}"
