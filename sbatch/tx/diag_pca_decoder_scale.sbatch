#!/bin/bash
#SBATCH --job-name=tx-pca-dec-scale
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=sbatch/logs/tx_pca_dec_scale_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Tied inverse PCA vs scalar / per-gene affine / full linear D_1 on HepG2 test.
#
#   sbatch sbatch/tx/diag_pca_decoder_scale.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

"${PYTHON_BIN}" experiments/tx/diag_pca_decoder_scale.py --device cuda "$@"
