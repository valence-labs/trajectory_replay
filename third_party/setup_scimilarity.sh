#!/usr/bin/env bash
# Isolated UV env for Genentech/SCimilarity (Python 3.12). Does not touch the
# repo root .venv (Python 3.13+).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/third_party/scimilarity"
PIN_FILE="${REPO_ROOT}/third_party/scimilarity.PIN.txt"
PYTHON="${SCIMILARITY_PYTHON:-3.12}"

if [[ ! -f "${PIN_FILE}" ]]; then
  echo "ERROR: missing pin file ${PIN_FILE}" >&2
  exit 1
fi
PIN="$(tr -d '[:space:]' < "${PIN_FILE}")"

mkdir -p "${DEST}"
if [[ -z "${UV_CACHE_DIR:-}" ]] || [[ ! -w "${UV_CACHE_DIR}" ]]; then
  export UV_CACHE_DIR="${REPO_ROOT}/.uv-cache"
  mkdir -p "${UV_CACHE_DIR}"
fi

if [[ ! -x "${DEST}/.venv/bin/python" ]]; then
  echo "Creating ${DEST}/.venv (Python ${PYTHON})"
  uv venv --python "${PYTHON}" "${DEST}/.venv"
fi

echo "Installing scimilarity==${PIN} (+ torch) into ${DEST}/.venv"
uv pip install --python "${DEST}/.venv/bin/python" "scimilarity==${PIN}" torch

echo
echo "OK. Encode with:"
echo "  STAGES=sci sbatch ${REPO_ROOT}/sbatch/tx/build_replogle.sbatch"
echo "Encoder weights: https://zenodo.org/records/10685499 (encoder.ckpt, gene_order.tsv, layer_sizes.json, label_ints.csv)"
