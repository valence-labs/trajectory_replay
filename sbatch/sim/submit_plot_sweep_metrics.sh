#!/usr/bin/env bash
# Submit sweep-metrics plotting to Slurm.
#
#   ./sbatch/sim/submit_plot_sweep_metrics.sh <sweep_dir> <output_dir>
#
# Example:
#   ./sbatch/sim/submit_plot_sweep_metrics.sh \
#     results_dir/simulation_results/sweep_ft_small_pert_effect \
#     results_dir/simulation_results/sweep_ft_small_pert_effect/metrics_plots

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <sweep_dir> <output_dir>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

SWEEP_DIR="$1"
OUTPUT_DIR="$2"
mkdir -p "${SBATCH_ROOT}/logs"

sbatch \
  --export="ALL,SIM_ROOT=${SIM_ROOT},SWEEP_DIR=${SWEEP_DIR},OUTPUT_DIR=${OUTPUT_DIR}" \
  --job-name="plot-sweep" \
  "${SCRIPT_DIR}/plot_sweep_metrics.sbatch"
