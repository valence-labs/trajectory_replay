#!/bin/bash
#SBATCH --job-name=se-pca-recon
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/se_pca_recon_%j.log

# PCA vs SE gene-space reconstruction + UMAPs (streamed metrics; UMAP subset).
#
# Replogle (defaults):
#   sbatch sbatch/tx/run_compare_se_pca_recon.sbatch
#
# PBMC (raise mem; SE may be .npy):
#   EXPR_H5AD=data_dir/pbmc_parse_with_splits.h5ad \
#   SE_H5AD=results_dir/arc_state/se/pbmc_parse_with_splits_se.npy \
#   SPLIT_COL=split3 \
#   OUT_DIR=results_dir/arc_state/pbmc_se_vs_pca_recon_pcs1000 \
#   sbatch --mem=256G --time=24:00:00 sbatch/tx/run_compare_se_pca_recon.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

: "${EXPR_H5AD:=${REPO_ROOT}/data_dir/replogle_with_splits.h5ad}"
: "${SE_H5AD:=${REPO_ROOT}/results_dir/arc_state/se/replogle_with_splits_se.h5ad}"
: "${N_PCS:=1000}"
: "${SPLIT_COL:=split}"
: "${OUT_DIR:=${REPO_ROOT}/results_dir/arc_state/se_vs_pca_recon_pcs${N_PCS}}"
: "${MAX_RIDGE_FIT:=100000}"
: "${MAX_PCA_FIT:=100000}"
: "${UMAP_MAX_CELLS:=20000}"
: "${DECODE_BATCH_SIZE:=8192}"

if [[ ! -f "${SE_H5AD}" ]]; then
  echo "ERROR: SE embeddings not found: ${SE_H5AD}" >&2
  echo "Wait for encode_se job to finish." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

echo "EXPR_H5AD=${EXPR_H5AD}"
echo "SE_H5AD=${SE_H5AD}"
echo "SPLIT_COL=${SPLIT_COL}"
echo "N_PCS=${N_PCS}"
echo "OUT_DIR=${OUT_DIR}"
echo "UMAP_MAX_CELLS=${UMAP_MAX_CELLS}"

"${PYTHON_BIN}" sbatch/tx/compare_se_pca_recon.py \
  --expr-h5ad "${EXPR_H5AD}" \
  --se-h5ad "${SE_H5AD}" \
  --split-col "${SPLIT_COL}" \
  --n-pcs "${N_PCS}" \
  --out-dir "${OUT_DIR}" \
  --max-ridge-fit "${MAX_RIDGE_FIT}" \
  --max-pca-fit "${MAX_PCA_FIT}" \
  --umap-max-cells "${UMAP_MAX_CELLS}" \
  --decode-batch-size "${DECODE_BATCH_SIZE}"
