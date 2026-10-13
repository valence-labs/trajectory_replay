#!/bin/bash
#SBATCH --job-name=plot-replay
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/plot_replay_seen_%j.log

# Plot seen-strata metrics vs replay fraction / buffer size for a replay sweep.
# Grid (must match submit_sweep_ft_replay.sh): vf+emb+ on ctx/pert/combo; vf−emb− on combo only.
#
#   sbatch --export=ALL,SWEEP_DIR=results_dir/sweep_ft_replay,OUTPUT_DIR=results_dir/sweep_ft_replay/replay_seen_plots \
#     sbatch/sim/plot_replay_seen_metrics.sbatch
#
# Or:
#   ./sbatch/sim/submit_plot_replay_seen_metrics.sh

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
SBATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

: "${SWEEP_DIR:?Set SWEEP_DIR (e.g. results_dir/sweep_ft_replay)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR (e.g. results_dir/sweep_ft_replay/replay_seen_plots)}"

extra_args=()
if [[ "${NO_PER_STRATUM:-0}" == "1" ]]; then
  extra_args+=(--no-per-stratum)
fi
if [[ "${LINEAR_Y:-0}" == "1" ]]; then
  extra_args+=(--linear-y)
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)"
echo "  SWEEP_DIR=${SWEEP_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"

if [[ ! -d "${SWEEP_DIR}" ]]; then
  echo "ERROR: sweep directory not found: ${SWEEP_DIR}" >&2
  exit 1
fi

uv sync
uv run python experiments/sim/plotting/plot_replay_seen_metrics.py "${SWEEP_DIR}" "${OUTPUT_DIR}" "${extra_args[@]}"

echo "Finished at $(date -Is)"
echo "Plots: ${OUTPUT_DIR}"
