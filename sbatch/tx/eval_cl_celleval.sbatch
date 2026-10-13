#!/bin/bash
#SBATCH --job-name=tx-cl-ce
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/tx_cl_celleval_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${OUT_DIR:?set OUT_DIR to the train_cl run directory}"
: "${DEVICE:=cuda}"
: "${HELDOUT_CURVE:=0}"
: "${SAVE_H5AD:=0}"
: "${TRACKS:=}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
EXTRA_ARGS=()
if [[ "${SAVE_H5AD}" == "1" ]]; then
  EXTRA_ARGS+=(--save-h5ad)
fi
if [[ -n "${TRACKS}" ]]; then
  TRACKS_NORM="${TRACKS//,/ }"
  TRACKS_NORM="${TRACKS_NORM//+/ }"
  read -r -a TRACK_ARGS <<< "${TRACKS_NORM}"
  EXTRA_ARGS+=(--tracks "${TRACK_ARGS[@]}")
fi

if [[ "${HELDOUT_CURVE}" != "1" ]]; then
  "${PYTHON_BIN}" experiments/tx/eval_cl_celleval.py \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"
fi
if [[ "${HELDOUT_CURVE}" != "0" ]]; then
  "${PYTHON_BIN}" experiments/tx/eval_cl_celleval.py \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --heldout-curve
fi
