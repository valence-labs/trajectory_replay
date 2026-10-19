#!/usr/bin/env bash
#SBATCH --job-name=px-representative
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/px_representative_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

srun uv run --no-sync python experiments/px/plot_representative_generations.py "$@"
