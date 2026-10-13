#!/bin/bash
#SBATCH --job-name=tx-oneshot-oracle
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tx_oneshot_oracle_%j.log
#SBATCH --partition=cpu

# Oracle Pearson-Δ for a one-shot run: decode observed embeddings, no ODE.
#
#   RUN_DIR=results_dir/tx_results/replogle/oneshot_sci_hepg2_panel6k_hvg2k_fresh_s42 \
#     sbatch sbatch/tx/diag_oneshot_decoder_oracle.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${RUN_DIR:=results_dir/tx_results/replogle/oneshot_sci_hepg2_panel6k_hvg2k_fresh_s42}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

echo "RUN_DIR=${RUN_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"

DEC="${RUN_DIR}/decoders/decoder_task00.npz"
if [[ "${WAIT_FOR_DECODER:-0}" == "1" ]]; then
  echo "waiting for ${DEC}"
  for _ in $(seq 1 360); do
    if [[ -f "${DEC}" ]]; then
      break
    fi
    sleep 30
  done
  if [[ ! -f "${DEC}" ]]; then
    echo "ERROR: decoder never appeared: ${DEC}" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" experiments/tx/diag_oneshot_decoder_oracle.py \
  --run-dir "${RUN_DIR}"
