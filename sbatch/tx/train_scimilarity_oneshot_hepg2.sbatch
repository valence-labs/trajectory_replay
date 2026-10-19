#!/bin/bash
#SBATCH --job-name=tx-sci-oneshot
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=sbatch/logs/tx_sci_oneshot_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# One-shot CFM on Replogle SCimilarity embeddings. Same trainer as SE.
# Encode first: STAGES=sci sbatch sbatch/tx/build_replogle.sbatch
#
#   sbatch sbatch/tx/train_scimilarity_oneshot_hepg2.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
export REPRESENTATION=scimilarity
export SE_PATH="${SE_PATH:-${REPO_ROOT}/data_dir/replogle/scimilarity.npy}"
export RUN_NAME="${RUN_NAME:-oneshot_sci_hepg2_panel6k_hvg2k_fresh_s42}"
if [[ ! -f "${SE_PATH}" ]]; then
  echo "ERROR: SCimilarity sidecar missing: ${SE_PATH}" >&2
  exit 1
fi
exec bash "${REPO_ROOT}/sbatch/tx/train_se_oneshot_hepg2.sbatch"
