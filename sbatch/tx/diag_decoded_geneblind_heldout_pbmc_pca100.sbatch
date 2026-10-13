#!/bin/bash
#SBATCH --job-name=tx-pbmc-pca100-ref
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/tx_pbmc_pca100_heldout_refs_%j.log

# PBMC B1/B3/B4/O1/C2 held-out references in decoded PCA-100 space.
#
#   sbatch sbatch/tx/diag_decoded_geneblind_heldout_pbmc_pca100.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/pbmc

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
: "${H5AD_PATH:=${REPO_ROOT}/data_dir/pbmc/adata.h5ad}"
: "${PCA_PATH:=${REPO_ROOT}/data_dir/pbmc/pca100_embedding.npy}"
: "${CONDITIONING:=donor_cell_type}"
: "${DECODER_DIR:=${REPO_ROOT}/results_dir/tx_results/pbmc/decoders_pca100_affine}"
: "${OUT_JSON:=${REPO_ROOT}/results_dir/tx_results/pbmc/diag_decoded_geneblind_heldout_pca100_${CONDITIONING}.json}"
: "${CAP:=0}"
: "${CONTROL_CAP:=0}"
: "${SEED:=42}"

echo "H5AD_PATH=${H5AD_PATH}"
echo "PCA_PATH=${PCA_PATH}"
echo "DECODER_DIR=${DECODER_DIR}"
echo "CONDITIONING=${CONDITIONING}"
echo "OUT_JSON=${OUT_JSON}"
echo "CAP=${CAP}"
echo "CONTROL_CAP=${CONTROL_CAP}"
echo "SEED=${SEED}"

"${PYTHON_BIN}" experiments/tx/diag_decoded_geneblind_heldout_pbmc.py \
  --h5ad-path "${H5AD_PATH}" \
  --pca-path "${PCA_PATH}" \
  --decoder-dir "${DECODER_DIR}" \
  --conditioning "${CONDITIONING}" \
  --cap "${CAP}" \
  --control-cap "${CONTROL_CAP}" \
  --seed "${SEED}" \
  --out-json "${OUT_JSON}"
