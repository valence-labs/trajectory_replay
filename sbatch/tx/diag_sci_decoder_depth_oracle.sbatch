#!/bin/bash
#SBATCH --job-name=tx-sci-dec-depth
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/tx_sci_dec_depth_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Oracle Pearson-Δ vs SCimilarity decoder depth (no CFM).
#
#   sbatch sbatch/tx/diag_sci_decoder_depth_oracle.sbatch
#   STANDARDIZE_LATENT=1 sbatch sbatch/tx/diag_sci_decoder_depth_oracle.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${SE_PATH:=${REPO_ROOT}/data_dir/replogle/scimilarity.npy}"
: "${SPECS:=linear 256x1 512x1 1024x1 1024x2 1024x3}"
: "${DEVICE:=cuda}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

extra=()
out="${OUT_JSON:-results_dir/tx_results/replogle/diag_sci_decoder_depth_oracle.json}"
if [[ "${STANDARDIZE_LATENT:-0}" == "1" ]]; then
  extra+=(--standardize-latent)
  out="${OUT_JSON:-results_dir/tx_results/replogle/diag_sci_decoder_depth_oracle_std.json}"
fi
if [[ "${RADIAL_SCALE_LATENT:-0}" == "1" ]]; then
  extra+=(--radial-scale-latent)
  out="${OUT_JSON:-results_dir/tx_results/replogle/diag_sci_decoder_depth_oracle_radial.json}"
fi

# shellcheck disable=SC2206
spec_arr=(${SPECS})
echo "SE_PATH=${SE_PATH} SPECS=${SPECS} STANDARDIZE_LATENT=${STANDARDIZE_LATENT:-0} RADIAL_SCALE_LATENT=${RADIAL_SCALE_LATENT:-0}"

"${PYTHON_BIN}" experiments/tx/diag_sci_decoder_depth_oracle.py \
  --se-path "${SE_PATH}" \
  --specs "${spec_arr[@]}" \
  --device "${DEVICE}" \
  --out-json "${out}" \
  "${extra[@]}"
