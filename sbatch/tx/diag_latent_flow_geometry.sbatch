#!/bin/bash
#SBATCH --job-name=tx-latent-geom
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/logs/tx_latent_geom_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Split a one-shot run's Pearson-Δ into flow (latent) and decoder halves.
#
#   RUN_DIR=results_dir/tx_results/replogle/oneshot_sci_hepg2_panel6k_hvg2k_fresh_s42 \
#     sbatch sbatch/tx/diag_latent_flow_geometry.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${RUN_DIR:=results_dir/tx_results/replogle/oneshot_sci_hepg2_panel6k_hvg2k_fresh_s42}"
: "${DEVICE:=cuda}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

echo "RUN_DIR=${RUN_DIR} DEVICE=${DEVICE}"

"${PYTHON_BIN}" experiments/tx/diag_latent_flow_geometry.py \
  --run-dir "${RUN_DIR}" \
  --device "${DEVICE}"
