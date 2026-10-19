#!/usr/bin/env bash
# Fit PBMC task-0 PCA (nested 512 → 50/100/256/512) then four oneshot CFMs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs data_dir/pbmc results_dir/tx_results/pbmc

FIT=$(sbatch --parsable --export=ALL,DATASET=pbmc,NESTED=1,SKIP_ORACLE=1 \
  sbatch/tx/fit_pca_task0_widths.sbatch)
echo "pca_fit job=${FIT}"

H5AD="${REPO_ROOT}/data_dir/pbmc_parse_with_splits.h5ad"
PCA_ROOT="${REPO_ROOT}/data_dir/pbmc"

for DIM in 50 100 256 512; do
  if [[ "${DIM}" == "256" ]]; then
    BASIS="${PCA_ROOT}/pca_basis.npz"
    EMB="${PCA_ROOT}/pca_embedding.npy"
  else
    BASIS="${PCA_ROOT}/pca${DIM}_basis.npz"
    EMB="${PCA_ROOT}/pca${DIM}_embedding.npy"
  fi
  RUN="oneshot_pca${DIM}_task0_pbmc_h1024_d4_st100000_s42"
  JOB=$(sbatch --parsable --dependency="afterok:${FIT}" --qos=YOUR_QOS --mem=256G \
    --export=ALL,DATASET=pbmc,SEED=42,TARGET_STEPS=100000,HIDDEN_DIM=1024,DEPTH=4,SKIP_CELLEVAL=1,PCA_DECODER=inverse,PRED_CONTROL=decoded,H5AD_PATH="${H5AD}",PCA_BASIS="${BASIS}",PCA_EMBEDDING="${EMB}",PCA_DIM=0,RUN_NAME="${RUN}" \
    sbatch/tx/train_pca_oneshot_hepg2.sbatch)
  echo "pca_dim=${DIM} oneshot job=${JOB} afterok:${FIT} run=${RUN}"
done
