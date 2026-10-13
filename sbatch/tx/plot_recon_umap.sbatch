#!/bin/bash
#SBATCH --job-name=sci-recon-umap
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=sbatch/logs/sci_recon_umap_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# True vs reconstructed HVG UMAPs for ridge / PCA / STATE / SCimilarity MLPs.
#
#   sbatch sbatch/tx/plot_recon_umap.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

"${PYTHON_BIN}" experiments/tx/plot_recon_umap.py --device cuda "$@"
