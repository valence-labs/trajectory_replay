#!/bin/bash
#SBATCH --job-name=tx-l1-validate
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/tx_l1_validate_%j.log
#SBATCH --partition=cpu

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/replogle

BASE="${REPO_ROOT}/results_dir/tx_results/replogle/standard_pca256_p0c0_st100000_ft50000_s42/cell_eval_minimal/final/fit/hepg2"
exec "${REPO_ROOT}/.venv/bin/python3" experiments/tx/validate_l1_against_celleval.py \
  --real-h5ad "${BASE}/adata_real.h5ad" \
  --pred-h5ad "${BASE}/adata_pred.h5ad" \
  --results "${BASE}/results.csv" \
  --out-json "${REPO_ROOT}/results_dir/tx_results/replogle/diag_l1_validation.json"
