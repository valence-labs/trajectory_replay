#!/usr/bin/env bash
#SBATCH --job-name=px-observed-grids
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/px_observed_grids_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

export UV_CACHE_DIR="${HOME}/.cache/uv"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

uv run --no-sync python experiments/px/plot_representative_observed.py "$@"
