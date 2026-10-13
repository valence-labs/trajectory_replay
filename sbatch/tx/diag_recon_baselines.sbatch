#!/bin/bash
#SBATCH --job-name=recon-baselines
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=sbatch/logs/recon_baselines_%j.log
#SBATCH --partition=cpu

# Sanity baselines (mean predictor, pool mean, PCA-256) for the recon bench.
#
#   sbatch sbatch/tx/diag_recon_baselines.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

"${PYTHON_BIN}" experiments/tx/diag_recon_baselines.py "$@"
