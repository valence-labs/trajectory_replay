#!/bin/bash
#SBATCH --job-name=tx-held-474
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/tx_heldout_intersection474_%j.log
#SBATCH --partition=cpu

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/replogle

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
: "${H5AD_PATH:=${REPO_ROOT}/data_dir/replogle/adata.h5ad}"
: "${OUT_JSON:=${REPO_ROOT}/results_dir/tx_results/replogle/diag_heldout_intersection474.json}"

echo "H5AD_PATH=${H5AD_PATH}"
echo "OUT_JSON=${OUT_JSON}"

"${PYTHON_BIN}" experiments/tx/score_heldout_intersection474.py \
  --h5ad-path "${H5AD_PATH}" \
  --out-json "${OUT_JSON}"
