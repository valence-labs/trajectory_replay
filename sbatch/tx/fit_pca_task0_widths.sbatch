#!/bin/bash
#SBATCH --job-name=tx-pca-widths
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=sbatch/logs/tx_pca_widths_%j.log
#SBATCH --partition=cpu
#SBATCH --gres=NONE

# Fit task-0 PCA widths. Nested = one SVD at max(WIDTHS), then slice.
#
#   DATASET=pbmc WIDTHS="50 100 256 512" NESTED=1 SKIP_ORACLE=1 \
#     sbatch sbatch/tx/fit_pca_task0_widths.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
: "${DATASET:=replogle}"
: "${NESTED:=0}"
: "${SKIP_ORACLE:=0}"
: "${SKIP_FIT:=0}"
: "${SLICE_FROM:=0}"
: "${PCA_METHOD:=auto}"
: "${PCA_BATCH_SIZE:=8192}"
: "${OUT_JSON:=}"
if [[ "${DATASET}" == "pbmc" ]]; then
  : "${WIDTHS:=50 100 256 512}"
  : "${NESTED:=1}"
  : "${SKIP_ORACLE:=1}"
else
  : "${WIDTHS:=512 1024 2048}"
fi

extra=()
if [[ "${NESTED}" == "1" ]]; then
  extra+=(--nested)
fi
if [[ "${SKIP_ORACLE}" == "1" ]]; then
  extra+=(--skip-oracle)
fi
if [[ "${SKIP_FIT}" == "1" ]]; then
  extra+=(--skip-fit)
fi
if [[ "${SLICE_FROM}" != "0" ]]; then
  extra+=(--slice-from "${SLICE_FROM}")
fi
if [[ -n "${OUT_JSON}" ]]; then
  extra+=(--out-json "${OUT_JSON}")
fi

echo "DATASET=${DATASET} WIDTHS=${WIDTHS} NESTED=${NESTED} SLICE_FROM=${SLICE_FROM} SKIP_ORACLE=${SKIP_ORACLE} PCA_METHOD=${PCA_METHOD} OUT_JSON=${OUT_JSON:-default}"

# shellcheck disable=SC2086
"${PYTHON_BIN}" experiments/tx/fit_pca_task0_widths.py \
  --dataset "${DATASET}" \
  --widths ${WIDTHS} \
  --pca-method "${PCA_METHOD}" \
  --pca-batch-size "${PCA_BATCH_SIZE}" \
  "${extra[@]}"
