#!/bin/bash
#SBATCH --job-name=sci-recon-bench
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=sbatch/logs/sci_recon_bench_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Linear vs MLP reconstruction of 2000 HVGs from SCimilarity (no CFM).
#
#   sbatch sbatch/tx/bench_scimilarity_recon.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${H5AD_PATH:=${REPO_ROOT}/data_dir/replogle/adata.h5ad}"
: "${SCI_PATH:=${REPO_ROOT}/data_dir/replogle/scimilarity.npy}"
: "${SE_PATH:=${REPO_ROOT}/data_dir/replogle/se.npy}"
: "${OUT_DIR:=${REPO_ROOT}/results_dir/tx_results/replogle/sci_recon_bench}"
: "${MAX_FIT_CELLS:=100000}"
: "${EPOCHS:=200}"
: "${MLP_SPECS:=256x1,512x1,1024x1,1024x2}"
: "${SMOKE:=0}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

extra=()
if [[ "${SMOKE}" == "1" ]]; then
  extra+=(--smoke)
fi
if [[ -n "${SE_PATH}" && -f "${SE_PATH}" ]]; then
  extra+=(--se-path "${SE_PATH}")
fi

echo "OUT_DIR=${OUT_DIR} MAX_FIT_CELLS=${MAX_FIT_CELLS} EPOCHS=${EPOCHS} MLP_SPECS=${MLP_SPECS}"

"${PYTHON_BIN}" experiments/tx/bench_scimilarity_recon.py \
  --h5ad-path "${H5AD_PATH}" \
  --sci-path "${SCI_PATH}" \
  --out-dir "${OUT_DIR}" \
  --max-fit-cells "${MAX_FIT_CELLS}" \
  --epochs "${EPOCHS}" \
  --mlp-specs "${MLP_SPECS}" \
  --device cuda \
  "${extra[@]}"
