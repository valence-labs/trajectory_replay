#!/bin/bash
#SBATCH --job-name=tx-pca-untied-d
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/logs/tx_pca_untied_dec_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Task-0 PCA encoder; tied inverse vs HepG2 / all-nontest linear readout.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
: "${DEVICE:=cuda}"

"${PYTHON_BIN}" experiments/tx/diag_pca_untied_decoder_oracle.py --device "${DEVICE}"
