#!/usr/bin/env bash
# Replogle sequential CL at PCA-50, matched to the decoded HepG2 baselines.
#
# Same objects as experiments/tx/tables/replogle_heldout_baselines_pca50.tex:
#   data_dir/replogle/adata.h5ad  (resolves to the affine-bank h5ad)
#   pca50_embedding.npy / pca50_basis.npz
#   results_dir/tx_results/replogle/decoders_pca50_affine
# Same eval protocol:
#   all observed pert cells, all controls, 500 generated cells per stratum,
#   pred_control=decoded, 100k T0 / 50k T1. λ-grid checkpoint ranking
#   uses K=200, matching the final PBMC campaign.
#
# Dry-run names:
#   DRY_RUN=1 ./sbatch/tx/submit_cl_pca50.sh
# First comparable seed (standard, no replay):
#   STRATEGIES="standard" SEEDS="42" ./sbatch/tx/submit_cl_pca50.sh
# Full sequential campaign at 50 PCs:
#   ./sbatch/tx/submit_cl_pca50.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export DATASETS="${DATASETS:-replogle}"
export REPRESENTATION=pca
export PCA_DIM=50
export LATER_TARGET_STEPS="${LATER_TARGET_STEPS:-50000}"
export TARGET_STEPS="${TARGET_STEPS:-100000}"
# Bank already exists and matches the baseline table; do not refit.
export PREPARE_DECODERS="${PREPARE_DECODERS:-0}"
export REQUIRE_EXISTING_DECODERS="${REQUIRE_EXISTING_DECODERS:-1}"
export PRED_CONTROL="${PRED_CONTROL:-decoded}"
export EVAL_MAX_CELLS_PER_STRATUM="${EVAL_MAX_CELLS_PER_STRATUM:-0}"
export EVAL_MAX_CONTROL_CELLS="${EVAL_MAX_CONTROL_CELLS:-0}"
export EVAL_GEN_PER_STRATUM="${EVAL_GEN_PER_STRATUM:-500}"
export LAMBDA_GEN_PER_STRATUM="${LAMBDA_GEN_PER_STRATUM:-200}"
export REPLAY_CONDITION_SAMPLING="${REPLAY_CONDITION_SAMPLING:-proportional}"
export DECODER_DIR="${DECODER_DIR:-${REPO_ROOT}/results_dir/tx_results/replogle/decoders_pca50_affine}"

exec "${REPO_ROOT}/sbatch/tx/submit_cl_campaign.sh" "$@"
