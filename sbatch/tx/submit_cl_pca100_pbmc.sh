#!/usr/bin/env bash
# PBMC sequential CL at PCA-100, using the already-fitted affine decoder bank.
#
#   data_dir/pbmc/adata.h5ad
#   pca100_embedding.npy / pca100_basis.npz
#   results_dir/tx_results/pbmc/decoders_pca100_affine
#
# Step budget: 300k T0 (~50 epochs on 6.16M train cells), 150k each later
# donor task. Run names include _st300000_ft150000.
# Final evaluation uses K=500 generated cells per
# (donor, cytokine, cell_type) stratum. λ-grid uses K=200 for checkpoint
# ranking only, matching the established PBMC campaign protocol.
#
# Does not submit prepare_cl_decoders. Training loads that bank and exits if
# any task decoder is missing or the manifest does not match.
#
# Dry-run names:
#   DRY_RUN=1 ./sbatch/tx/submit_cl_pca100_pbmc.sh
# First comparable seed (standard, no replay):
#   STRATEGIES="standard" SEEDS="42" ./sbatch/tx/submit_cl_pca100_pbmc.sh
# Full sequential campaign at 100 PCs:
#   ./sbatch/tx/submit_cl_pca100_pbmc.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export DATASETS="${DATASETS:-pbmc}"
export REPRESENTATION=pca
export PCA_DIM=100
export TARGET_STEPS="${TARGET_STEPS:-300000}"
export LATER_TARGET_STEPS="${LATER_TARGET_STEPS:-150000}"
export PREPARE_DECODERS="${PREPARE_DECODERS:-0}"
export REQUIRE_EXISTING_DECODERS="${REQUIRE_EXISTING_DECODERS:-1}"
export PRED_CONTROL="${PRED_CONTROL:-decoded}"
export EVAL_MAX_CELLS_PER_STRATUM="${EVAL_MAX_CELLS_PER_STRATUM:-0}"
export EVAL_MAX_CONTROL_CELLS="${EVAL_MAX_CONTROL_CELLS:-0}"
export EVAL_GEN_PER_STRATUM="${EVAL_GEN_PER_STRATUM:-500}"
export LAMBDA_GEN_PER_STRATUM="${LAMBDA_GEN_PER_STRATUM:-200}"
export REPLAY_CONDITION_SAMPLING="${REPLAY_CONDITION_SAMPLING:-proportional}"
# Pearson heldout curve stays in the train job. Final Pearson and Cell-Eval
# are stream + heldout only (fit/forget duplicate stream). Split stream by
# task_id for T0 vs later-donor table numbers. Re-enable the Cell-Eval curve
# with CELLEVAL_HELDOUT_CURVE=both; all four tracks with EVAL_FINAL_TRACKS
# and TRACKS unset.
export CELLEVAL_HELDOUT_CURVE="${CELLEVAL_HELDOUT_CURVE:-0}"
export EVAL_FINAL_TRACKS="${EVAL_FINAL_TRACKS:-stream+heldout}"
export TRACKS="${TRACKS:-stream+heldout}"
export DECODER_DIR="${DECODER_DIR:-${REPO_ROOT}/results_dir/tx_results/pbmc/decoders_pca100_affine}"

if [[ ! -f "${DECODER_DIR}/manifest.json" ]]; then
  echo "Missing decoder bank manifest: ${DECODER_DIR}/manifest.json" >&2
  exit 2
fi

exec "${REPO_ROOT}/sbatch/tx/submit_cl_campaign.sh" "$@"
