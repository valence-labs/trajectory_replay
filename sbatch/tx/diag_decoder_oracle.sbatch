#!/bin/bash
#SBATCH --job-name=tx-cl-dec-oracle
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tx_cl_dec_oracle_%j.log
#SBATCH --partition=YOUR_CPU_PARTITION

# Oracle Pearson-Δ from decoding observed SE (no ODE / GPU).
#
#   sbatch sbatch/tx/diag_decoder_oracle.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${RUN_DIR:=results_dir/tx_results/replogle/real_grid_st100000_s42}"
: "${OUT_JSON:=results_dir/tx_results/replogle/diag_decoder_oracle_fit_heldout.json}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

echo "RUN_DIR=${RUN_DIR}"
echo "OUT_JSON=${OUT_JSON}"
echo "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" experiments/tx/diag_decoder_oracle.py \
  --run-dir "${RUN_DIR}" \
  --out-json "${OUT_JSON}"
