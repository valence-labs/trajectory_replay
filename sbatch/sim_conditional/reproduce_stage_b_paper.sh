#!/usr/bin/env bash
# Canonical reproduction of the Stage B additive-shift paper experiment.
#
# Pins every setting used for the published figures so no env overrides are
# needed: c6_p4 (dim=2), seeds 10-50, batch 1024, lr 3e-4, pretrain 1000 ep,
# finetune 2000 ep, Euler ODE step 0.05, no gradient clipping.
#
# Steps: build data (if missing) -> pretrain + λ grid via the throttled
# submitter -> aggregate λ figures.
#
# Run in tmux (the submitter blocks while batches finish):
#   tmux new -s stage_b
#   ./sbatch/sim_conditional/reproduce_stage_b_paper.sh
#
# Dry run (no sbatch, no data build):
#   DRY_RUN=1 ./sbatch/sim_conditional/reproduce_stage_b_paper.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SIM_ROOT}"

DRY_RUN="${DRY_RUN:-0}"

# --- Paper configuration (do not change to reproduce published figures) ---
export STAGE_B_DIM=2
export N_CONTEXTS=6
export N_PERTURBATIONS=4
export SEEDS="10 20 30 40 50"
export LAMBDAS="0.05 0.1 0.25 0.5 0.75 0.9 0.95"
export STRATEGIES="naive generative trajectory"
export PRETRAIN_EPOCHS=1000
export FT_EPOCHS=2000
export BATCH_SIZE=1024
export LR=3e-4
export DEVICE="${DEVICE:-cuda}"
export TRAJ_KAPPA=none
export DRY_RUN

CONFIG_TAG="c${N_CONTEXTS}_p${N_PERTURBATIONS}"
DATA_ROOT="${SIM_ROOT}/results_dir/sim_conditional/stage_b_shift/data/dim${STAGE_B_DIM}/${CONFIG_TAG}"

echo "=== Stage B paper reproduction (${CONFIG_TAG}, dim=${STAGE_B_DIM}) ==="

# --- 1. Data (defaults already encode the paper geometry) ---
if [[ -f "${DATA_ROOT}/meta.json" ]]; then
  echo "Reuse existing data at ${DATA_ROOT}"
elif [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY: would build data at ${DATA_ROOT}"
else
  echo "Building data at ${DATA_ROOT}"
  uv run python experiments/sim_conditional/stage_b_shift/build_datasets.py \
    --dim "${STAGE_B_DIM}" \
    --n-contexts "${N_CONTEXTS}" \
    --n-perturbations "${N_PERTURBATIONS}"
  uv run python experiments/sim_conditional/stage_b_shift/plot_data.py \
    --dim "${STAGE_B_DIM}" \
    --n-contexts "${N_CONTEXTS}" \
    --n-perturbations "${N_PERTURBATIONS}"
fi

# --- 2. Pretrain + λ grid (skips runs that already have metrics.json) ---
"${SCRIPT_DIR}/submit_stage_b_throttled.sh"

# --- 3. Aggregate figures (energy = main text, mean_l2 = appendix) ---
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY: would render λ figures"
  exit 0
fi

for metric in energy mean_l2; do
  uv run python experiments/sim_conditional/stage_b_shift/plot_lambda_fit_forget.py \
    --dim "${STAGE_B_DIM}" \
    --n-contexts "${N_CONTEXTS}" \
    --n-perturbations "${N_PERTURBATIONS}" \
    --epochs "${FT_EPOCHS}" \
    --metric "${metric}"
done

echo "=== Stage B reproduction complete ==="
