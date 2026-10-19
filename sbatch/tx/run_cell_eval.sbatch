#!/bin/bash
#SBATCH --job-name=tx-cell-eval
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=sbatch/logs/tx_cell_eval_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Post-hoc Cell-Eval on one tx run (val/test by default from metrics.json).
#
#   sbatch --export=ALL,TX_RESULTS_BUCKET=pbmc_split3_state,RUN_PATH=pbmc_split3_pretrain \
#     sbatch/tx/run_cell_eval.sbatch
#
# Override: EVAL_PHASES=val,test PROFILE=minimal DEVICE=cuda SBATCH_MEM=128G

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
# shellcheck source=sbatch/tx/tx_results_bucket.env.sh
source "${SIM_ROOT}/sbatch/tx/tx_results_bucket.env.sh"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${RUN_PATH:?Set RUN_PATH to bucket-relative run dir (e.g. pbmc_split3_pretrain)}"
: "${DEVICE:=cuda}"
: "${PROFILE:=full}"
: "${EVAL_PHASES:=auto}"
: "${CONTEXT:=}"
: "${NUM_THREADS:=-1}"
: "${USE_OBSERVED_CONTROLS_ONLY:=1}"
: "${USE_RAW_X:=1}"

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  TX_RESULTS_BUCKET=${TX_RESULTS_BUCKET}  RUN_PATH=${RUN_PATH}"
echo "  PROFILE=${PROFILE}  DEVICE=${DEVICE}  EVAL_PHASES=${EVAL_PHASES:-<metrics>}  CONTEXT=${CONTEXT:-all}"
echo "  USE_OBSERVED_CONTROLS_ONLY=${USE_OBSERVED_CONTROLS_ONLY}  USE_RAW_X=${USE_RAW_X}"

# Install both extras so concurrent vcb jobs do not strip cell-eval from the shared venv.
uv sync --extra cell-eval --extra vcb
uv run python -c "from cell_eval import MetricsEvaluator"

eval_args=(experiments/tx/run_cell_eval.py "${RUN_PATH}" --device "${DEVICE}" --profile "${PROFILE}" --num-threads "${NUM_THREADS}")
if [[ -n "${EVAL_PHASES}" ]]; then
  eval_args+=(--eval-phases "${EVAL_PHASES}")
fi
if [[ -n "${CONTEXT}" ]]; then
  eval_args+=(--context "${CONTEXT}")
fi
if [[ "${USE_OBSERVED_CONTROLS_ONLY}" == "0" ]]; then
  eval_args+=(--no-use-observed-controls-only)
fi
if [[ "${USE_RAW_X}" == "0" ]]; then
  eval_args+=(--no-use-raw-x)
fi
if [[ -n "${AGGREGATE_COVARIATE:-}" ]]; then
  eval_args+=(--aggregate-covariate "${AGGREGATE_COVARIATE}")
fi
if [[ "${SKIP_EXISTING_PHASES:-0}" == "1" ]]; then
  eval_args+=(--skip-existing-phases)
fi

uv run python "${eval_args[@]}"

echo "Finished at $(date -Is)"
