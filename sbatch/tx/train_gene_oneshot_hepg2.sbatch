#!/bin/bash
#SBATCH --job-name=tx-gene-oneshot
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=sbatch/logs/tx_gene_oneshot_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# One-shot CFM on Replogle gene expression (all non-test cells), HepG2 eval.
#
#   sbatch sbatch/tx/train_gene_oneshot_hepg2.sbatch
#   TARGET_STEPS=100000 SEED=42 sbatch sbatch/tx/train_gene_oneshot_hepg2.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results

: "${SEED:=42}"
: "${TARGET_STEPS:=100000}"
: "${HIDDEN_DIM:=1024}"
: "${DEPTH:=4}"
: "${DEVICE:=cuda}"
: "${H5AD_PATH:=${REPO_ROOT}/data_dir/replogle/adata.h5ad}"
: "${RUN_NAME:=oneshot_gene_hepg2_h${HIDDEN_DIM}_d${DEPTH}_st${TARGET_STEPS}_s${SEED}}"
: "${SMOKE:=0}"
: "${SKIP_TRAIN:=0}"
: "${SKIP_CELLEVAL:=0}"
: "${SYNC_ENV:=0}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

extra=()
if [[ "${SMOKE}" == "1" ]]; then
  extra+=(--smoke)
fi
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  extra+=(--skip-train)
fi
if [[ "${SKIP_CELLEVAL}" == "1" ]]; then
  extra+=(--skip-celleval)
fi

echo "RUN_NAME=${RUN_NAME} TARGET_STEPS=${TARGET_STEPS} SEED=${SEED} HIDDEN_DIM=${HIDDEN_DIM} DEPTH=${DEPTH} H5AD_PATH=${H5AD_PATH}"

if [[ "${SYNC_ENV}" == "1" ]]; then
  uv sync --extra cell-eval --extra vcb
fi
"${PYTHON_BIN}" -c "from cell_eval import MetricsEvaluator"

"${PYTHON_BIN}" experiments/tx/train_gene_oneshot_hepg2.py \
  --dataset replogle \
  --h5ad-path "${H5AD_PATH}" \
  --run-name "${RUN_NAME}" \
  --target-steps "${TARGET_STEPS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${extra[@]}"
