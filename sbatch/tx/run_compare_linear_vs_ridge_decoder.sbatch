#!/bin/bash
#SBATCH --job-name=lin-vs-ridge
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=sbatch/logs/lin_vs_ridge_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results

: "${DATASET:=replogle}"
: "${MAX_FIT:=100000}"
: "${EPOCHS:=200}"
: "${LR:=0.001}"
: "${BATCH_SIZE:=8192}"
: "${DEVICE:=auto}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
OUT="results_dir/tx_results/${DATASET}_linear_vs_ridge.json"

"${PYTHON_BIN}" sbatch/tx/compare_linear_vs_ridge_decoder.py \
  --dataset "${DATASET}" \
  --max-fit "${MAX_FIT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --out-json "${OUT}"
