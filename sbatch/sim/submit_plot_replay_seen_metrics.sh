#!/usr/bin/env bash
# Submit replay seen-strata plotting to Slurm.
#
#   ./sbatch/sim/submit_plot_replay_seen_metrics.sh
#   ./sbatch/sim/submit_plot_replay_seen_metrics.sh <sweep_dir> <output_dir>
#
# Defaults:
#   SWEEP_DIR=results_dir/sweep_ft_replay
#   OUTPUT_DIR=results_dir/sweep_ft_replay/replay_seen_plots
#
# Y-axis: log scale for energy panels by default. Revert with:
#   LINEAR_Y=1 ./sbatch/sim/submit_plot_replay_seen_metrics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

SWEEP_DIR="${1:-results_dir/simulation_results/sweep_ft_replay}"
OUTPUT_DIR="${2:-${SWEEP_DIR}/replay_seen_plots}"
mkdir -p "${SBATCH_ROOT}/logs"

sbatch \
  --export="ALL,SIM_ROOT=${SIM_ROOT},SWEEP_DIR=${SWEEP_DIR},OUTPUT_DIR=${OUTPUT_DIR},NO_PER_STRATUM=${NO_PER_STRATUM:-0},LINEAR_Y=${LINEAR_Y:-0}" \
  --job-name="plot-replay" \
  "${SCRIPT_DIR}/plot_replay_seen_metrics.sbatch"

echo "Submitted plot job for ${SWEEP_DIR} → ${OUTPUT_DIR}"
