#!/bin/bash
#SBATCH --job-name=tx-std-dec-recon
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/logs/tx_std_dec_recon_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Raw vs whitened latent decoder reconstruction (no CFM).
#
#   SE_PATH=data_dir/replogle/scimilarity.npy REPRESENTATION=scimilarity \
#     sbatch sbatch/tx/diag_std_decoder_recon.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${REPRESENTATION:=scimilarity}"
if [[ "${REPRESENTATION}" == "scimilarity" ]]; then
  : "${SE_PATH:=${REPO_ROOT}/data_dir/replogle/scimilarity.npy}"
else
  : "${SE_PATH:=${REPO_ROOT}/data_dir/replogle/se.npy}"
fi
: "${N_FIT:=80000}"
: "${N_EVAL:=15000}"
: "${DECODER_EPOCHS:=200}"
: "${DEVICE:=cuda}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

echo "REPRESENTATION=${REPRESENTATION} SE_PATH=${SE_PATH} N_FIT=${N_FIT} DECODER_EPOCHS=${DECODER_EPOCHS}"

"${PYTHON_BIN}" experiments/tx/diag_std_decoder_recon.py \
  --se-path "${SE_PATH}" \
  --representation "${REPRESENTATION}" \
  --n-fit "${N_FIT}" \
  --n-eval "${N_EVAL}" \
  --decoder-epochs "${DECODER_EPOCHS}" \
  --device "${DEVICE}" \
  --out-json "results_dir/tx_results/replogle/diag_std_decoder_${REPRESENTATION}.json"
