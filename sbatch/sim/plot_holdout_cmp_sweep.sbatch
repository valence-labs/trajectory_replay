#!/bin/bash
#SBATCH --job-name=plot-holdout-cmp
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/plot_holdout_cmp_%j.log

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
SBATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

: "${SWEEP_DIR:?Set SWEEP_DIR (e.g. results_dir/sweep_holdout_cmp)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR (e.g. results_dir/sweep_holdout_cmp/metrics_plots)}"

echo "Job ${SLURM_JOB_ID:-local}  SWEEP_DIR=${SWEEP_DIR}  OUTPUT_DIR=${OUTPUT_DIR}"

uv sync
uv run python experiments/sim/plotting/plot_holdout_cmp_sweep.py "${SWEEP_DIR}" "${OUTPUT_DIR}"

echo "Finished at $(date -Is)"
