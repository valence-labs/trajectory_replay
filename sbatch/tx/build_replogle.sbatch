#!/bin/bash
#SBATCH --job-name=build-replogle
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/build_replogle_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Build every Replogle CL input from data_dir/replogle/replogle.h5ad.
#
#   sbatch sbatch/tx/build_replogle.sbatch                       # all stages
#   STAGES=adata,pca sbatch --partition=cpu --gres=NONE \
#     sbatch/tx/build_replogle.sbatch                            # CPU-only stages
#
# Stages are idempotent: existing outputs are skipped unless FORCE=1. The GPU is
# only needed for the se / sci encodes.
#
# Env overrides:
#   STAGES    — comma list of adata,pca,se,sci (default: all)
#   OUT_DIR   — dump folder (default: data_dir/replogle)
#   N_COMPS   — PCA components (default: 256 = unlabeled pca_*). Other widths
#               write pca{d}_basis.npz / pca{d}_embedding.npy and leave 256 alone.
#   FORCE     — set to 1 to rebuild stages whose outputs exist

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

: "${STAGES:=all}"
: "${OUT_DIR:=${REPO_ROOT}/data_dir/replogle}"
: "${N_COMPS:=256}"
: "${FORCE:=0}"

ARGS=(--out-dir "${OUT_DIR}" --stages "${STAGES}" --n-comps "${N_COMPS}")
if [[ "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "${SCI_SKIP_L2:-0}" == "1" ]]; then
  ARGS+=(--sci-skip-l2-normalize)
fi

echo "STAGES=${STAGES} OUT_DIR=${OUT_DIR} N_COMPS=${N_COMPS} FORCE=${FORCE} SCI_SKIP_L2=${SCI_SKIP_L2:-0}"
exec .venv/bin/python scripts/build_replogle.py "${ARGS[@]}"
