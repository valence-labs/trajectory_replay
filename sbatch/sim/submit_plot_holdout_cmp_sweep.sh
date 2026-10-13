#!/usr/bin/env bash
# Plot aggregated metrics for the hold-out comparison sweep.
#
#   ./sbatch/sim/submit_plot_holdout_cmp_sweep.sh
#   ./sbatch/sim/submit_plot_holdout_cmp_sweep.sh results_dir/sweep_holdout_cmp results_dir/sweep_holdout_cmp/metrics_plots

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

SWEEP_DIR="${1:-results_dir/simulation_results/sweep_holdout_cmp}"
OUTPUT_DIR="${2:-${SWEEP_DIR}/metrics_plots}"
mkdir -p "${SBATCH_ROOT}/logs"

sbatch \
  --export="ALL,SIM_ROOT=${SIM_ROOT},SWEEP_DIR=${SWEEP_DIR},OUTPUT_DIR=${OUTPUT_DIR}" \
  --job-name="plot-holdout-cmp" \
  "${SCRIPT_DIR}/plot_holdout_cmp_sweep.sbatch"

echo "Plot job submitted. Output: ${OUTPUT_DIR}"
