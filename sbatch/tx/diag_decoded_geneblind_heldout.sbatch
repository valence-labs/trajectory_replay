#!/bin/bash
#SBATCH --job-name=tx-decoded-blind
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tx_decoded_geneblind_%j.log
#SBATCH --partition=cpu

# Gene-blind / T0-transfer baselines on decoded PCA means (HepG2 affine D1).
#
#   sbatch sbatch/tx/diag_decoded_geneblind_heldout.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/replogle

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
: "${H5AD_PATH:=${REPO_ROOT}/data_dir/replogle/adata.h5ad}"
: "${PCA_PATH:=${REPO_ROOT}/data_dir/replogle/pca_embedding.npy}"
: "${DECODER_DIR:=${REPO_ROOT}/results_dir/tx_results/replogle/decoders_pca256_affine}"
: "${OUT_JSON:=${REPO_ROOT}/results_dir/tx_results/replogle/diag_decoded_geneblind_heldout.json}"
: "${CAP:=0}"
: "${CONTROL_CAP:=0}"

echo "H5AD_PATH=${H5AD_PATH}"
echo "PCA_PATH=${PCA_PATH}"
echo "DECODER_DIR=${DECODER_DIR}"
echo "OUT_JSON=${OUT_JSON}"
echo "CAP=${CAP}"
echo "CONTROL_CAP=${CONTROL_CAP}"

"${PYTHON_BIN}" experiments/tx/diag_decoded_geneblind_heldout.py \
  --h5ad-path "${H5AD_PATH}" \
  --pca-path "${PCA_PATH}" \
  --decoder-dir "${DECODER_DIR}" \
  --cap "${CAP}" \
  --control-cap "${CONTROL_CAP}" \
  --out-json "${OUT_JSON}"
