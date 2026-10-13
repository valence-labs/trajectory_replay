#!/bin/bash
#SBATCH --job-name=tx-cl-dec
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/tx_cl_decoders_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results

: "${DATASET:?set DATASET to pbmc or replogle}"
: "${REPRESENTATION:=se}"
: "${DEVICE:=cuda}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

extra=()
if [[ -n "${H5AD_PATH:-}" ]]; then
  extra+=(--h5ad-path "${H5AD_PATH}")
fi
if [[ -n "${SE_PATH:-}" ]]; then
  extra+=(--se-path "${SE_PATH}")
fi
if [[ -n "${PCA_BASIS:-}" ]]; then
  extra+=(--pca-basis "${PCA_BASIS}")
fi
if [[ "${REPRESENTATION}" == "pca" ]]; then
  # Keep in lockstep with DEFAULT_PCA_DIM in tx_preprocess.py.
  : "${PCA_DIM:=256}"
  extra+=(--pca-dim "${PCA_DIM}")
fi
if [[ -n "${DECODER_DIR:-}" ]]; then
  extra+=(--decoder-dir "${DECODER_DIR}")
fi
if [[ "${REQUIRE_EXISTING_DECODERS:-0}" == "1" ]]; then
  extra+=(--require-existing-decoders)
fi

"${PYTHON_BIN}" experiments/tx/prepare_cl_decoders.py \
  --dataset "${DATASET}" \
  --representation "${REPRESENTATION}" \
  --device "${DEVICE}" \
  "${extra[@]}"
