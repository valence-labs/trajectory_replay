#!/bin/bash
#SBATCH --job-name=tx-geneblind-held
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tx_geneblind_heldout_%j.log
#SBATCH --partition=YOUR_CPU_PARTITION

# Training-available gene-blind Pearson-Δ and L1-discrimination on HepG2 heldout.
#
#   sbatch sbatch/tx/diag_geneblind_heldout.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/replogle

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
: "${H5AD_PATH:=${REPO_ROOT}/data_dir/replogle_with_splits.h5ad}"
: "${OUT_JSON:=${REPO_ROOT}/results_dir/tx_results/replogle/diag_geneblind_heldout.json}"

echo "H5AD_PATH=${H5AD_PATH}"
echo "OUT_JSON=${OUT_JSON}"
echo "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/tx/diag_geneblind_heldout.py \
  --h5ad-path "${H5AD_PATH}" \
  --out-json "${OUT_JSON}"
