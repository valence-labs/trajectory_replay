#!/bin/bash
#SBATCH --job-name=tx-cl-env
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/tx_cl_env_%j.log
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --gres=NONE

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

# Sync the shared environment once before the campaign. Running this from every
# GPU job concurrently can contend on or partially rewrite the same .venv.
uv sync --extra cell-eval --extra vcb
"${REPO_ROOT}/.venv/bin/python3" -c \
  "from cell_eval import MetricsEvaluator; import torch"
