#!/bin/bash
#SBATCH --job-name=plot-sweep
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/plot_sweep_%j.log

# Plot metrics for a finetuning sweep directory.
#
# Usage:
#   sbatch --export=ALL,SWEEP_DIR=results_dir/sweep_ft_small_pert_effect,OUTPUT_DIR=results_dir/sweep_ft_small_pert_effect/metrics_plots sbatch/sim/plot_sweep_metrics.sbatch
#
# Or via helper:
#   ./sbatch/sim/submit_plot_sweep_metrics.sh results_dir/sweep_ft_small_pert_effect results_dir/sweep_ft_small_pert_effect/metrics_plots

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
SBATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

: "${SWEEP_DIR:?Set SWEEP_DIR to the finetuning sweep folder (e.g. results_dir/sweep_ft_small_pert_effect)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to where plots should be written}"

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)"
echo "  SWEEP_DIR=${SWEEP_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"

if [[ ! -d "${SWEEP_DIR}" ]]; then
  echo "ERROR: sweep directory not found: ${SWEEP_DIR}" >&2
  exit 1
fi

uv sync
uv run python experiments/sim/plotting/plot_sweep_metrics.py "${SWEEP_DIR}" "${OUTPUT_DIR}"

echo "Finished at $(date -Is)"
echo "Plots: ${OUTPUT_DIR}"
