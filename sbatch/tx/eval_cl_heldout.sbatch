#!/bin/bash
#SBATCH --job-name=tx-cl-ho
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/tx_cl_heldout_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Sequential heldout curve from a finished train_cl run.
#
#   sbatch --export=ALL,OUT_DIR=results_dir/tx_results/replogle/generative_equal_s42 \
#     sbatch/tx/eval_cl_heldout.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${OUT_DIR:?set OUT_DIR to the train_cl run directory}"
: "${DEVICE:=cuda}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

"${PYTHON_BIN}" experiments/tx/eval_cl_heldout.py \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}"
