#!/bin/bash
#SBATCH --job-name=tx-pca-oneshot
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=sbatch/logs/tx_pca_oneshot_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# One-shot CFM on Replogle PCA (all non-test cells), HepG2 eval.
#
# PCA_BASIS/PCA_EMBEDDING select the task-0 basis from build_replogle.py, which
# excludes HepG2 from the fit. Without them the dump's X_pca is used, which is
# fit on train+finetune and so encodes HepG2 (oneshot-only).
#
#   sbatch sbatch/tx/train_pca_oneshot_hepg2.sbatch
#   TARGET_STEPS=100000 SEED=42 sbatch sbatch/tx/train_pca_oneshot_hepg2.sbatch
#   PCA_BASIS=data_dir/replogle_pca_task0_basis.npz \
#     PCA_EMBEDDING=data_dir/replogle_pca_task0_embedding.npy \
#     sbatch sbatch/tx/train_pca_oneshot_hepg2.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results

: "${DATASET:=replogle}"
: "${SEED:=42}"
: "${TARGET_STEPS:=100000}"
: "${EPOCHS:=100}"
: "${HIDDEN_DIM:=1024}"
: "${DEPTH:=4}"
: "${COND_EMB_DIM:=128}"
: "${CONDITIONING_MODE:=concat}"
: "${DEVICE:=cuda}"
: "${H5AD_PATH:=${REPO_ROOT}/data_dir/replogle/adata.h5ad}"
: "${RUN_NAME:=oneshot_pca_hepg2_h${HIDDEN_DIM}_d${DEPTH}_st${TARGET_STEPS}_s${SEED}}"
: "${SMOKE:=0}"
: "${SKIP_TRAIN:=0}"
: "${SKIP_CELLEVAL:=0}"
: "${ODE_METHOD:=euler}"
: "${PCA_BASIS:=${REPO_ROOT}/data_dir/replogle/pca_basis.npz}"
: "${PCA_EMBEDDING:=${REPO_ROOT}/data_dir/replogle/pca_embedding.npy}"
: "${PRED_CONTROL:=decoded}"
: "${SYNC_ENV:=0}"
: "${PCA_DECODER:=inverse}"
: "${INIT_CKPT:=}"
: "${DECODER_HIDDEN:=0}"
: "${DECODER_DEPTH:=1}"
: "${PCA_DIM:=0}"
: "${DECODER_DIR:=}"
: "${CL_TRACKS:=0}"
: "${FLOW_TYPE:=n2d}"
: "${D2D_EXCLUDE_IDENTITY_TARGETS:=0}"
: "${CONDITIONING:=donor_cell_type}"
: "${CELL_EVAL_GROUPING:=condition}"
: "${PEARSON_GROUPING:=condition}"
: "${EVAL_MAX_CELLS_PER_STRATUM:=500}"
: "${EVAL_GEN_PER_STRATUM:=500}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

extra=()
if [[ -n "${PCA_BASIS}" || -n "${PCA_EMBEDDING}" ]]; then
  extra+=(--pca-basis "${PCA_BASIS}" --pca-embedding "${PCA_EMBEDDING}")
fi
if [[ "${SMOKE}" == "1" ]]; then
  extra+=(--smoke)
fi
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  extra+=(--skip-train)
fi
if [[ "${SKIP_CELLEVAL}" == "1" ]]; then
  extra+=(--skip-celleval)
fi
if [[ "${PRED_CONTROL}" != "decoded" ]]; then
  extra+=(--pred-control "${PRED_CONTROL}")
fi
if [[ -n "${INIT_CKPT}" ]]; then
  extra+=(--init-ckpt "${INIT_CKPT}")
fi
if [[ "${PCA_DECODER}" != "inverse" ]]; then
  extra+=(--pca-decoder "${PCA_DECODER}")
fi
if [[ "${PCA_DIM}" != "0" ]]; then
  extra+=(--pca-dim "${PCA_DIM}")
fi
if [[ -n "${DECODER_DIR}" ]]; then
  extra+=(--decoder-dir "${DECODER_DIR}")
fi
if [[ "${CL_TRACKS}" == "1" ]]; then
  extra+=(--cl-tracks)
fi
if [[ "${FLOW_TYPE}" != "n2d" ]]; then
  extra+=(--flow-type "${FLOW_TYPE}")
fi
if [[ "${D2D_EXCLUDE_IDENTITY_TARGETS}" == "1" ]]; then
  extra+=(--d2d-exclude-identity-targets)
fi
extra+=(--decoder-hidden "${DECODER_HIDDEN}" --decoder-depth "${DECODER_DEPTH}")

echo "DATASET=${DATASET} RUN_NAME=${RUN_NAME} FLOW_TYPE=${FLOW_TYPE} CONDITIONING=${CONDITIONING} CONDITIONING_MODE=${CONDITIONING_MODE} CELL_EVAL_GROUPING=${CELL_EVAL_GROUPING} PEARSON_GROUPING=${PEARSON_GROUPING} D2D_EXCLUDE_IDENTITY_TARGETS=${D2D_EXCLUDE_IDENTITY_TARGETS} TARGET_STEPS=${TARGET_STEPS} EPOCHS=${EPOCHS} SEED=${SEED} HIDDEN_DIM=${HIDDEN_DIM} DEPTH=${DEPTH} COND_EMB_DIM=${COND_EMB_DIM} H5AD_PATH=${H5AD_PATH} ODE_METHOD=${ODE_METHOD} SKIP_TRAIN=${SKIP_TRAIN} PRED_CONTROL=${PRED_CONTROL} PCA_BASIS=${PCA_BASIS} PCA_EMBEDDING=${PCA_EMBEDDING} PCA_DECODER=${PCA_DECODER} INIT_CKPT=${INIT_CKPT} PCA_DIM=${PCA_DIM} DECODER_DIR=${DECODER_DIR} CL_TRACKS=${CL_TRACKS} EVAL_MAX_CELLS_PER_STRATUM=${EVAL_MAX_CELLS_PER_STRATUM} EVAL_GEN_PER_STRATUM=${EVAL_GEN_PER_STRATUM}"

if [[ "${SYNC_ENV}" == "1" ]]; then
  uv sync --extra cell-eval --extra vcb
fi
"${PYTHON_BIN}" -c "from cell_eval import MetricsEvaluator"

"${PYTHON_BIN}" experiments/tx/train_pca_oneshot_hepg2.py \
  --dataset "${DATASET}" \
  --h5ad-path "${H5AD_PATH}" \
  --run-name "${RUN_NAME}" \
  --target-steps "${TARGET_STEPS}" \
  --epochs "${EPOCHS}" \
  --conditioning "${CONDITIONING}" \
  --conditioning-mode "${CONDITIONING_MODE}" \
  --cell-eval-grouping "${CELL_EVAL_GROUPING}" \
  --pearson-grouping "${PEARSON_GROUPING}" \
  --cond-emb-dim "${COND_EMB_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --ode-method "${ODE_METHOD}" \
  --eval-max-cells-per-stratum "${EVAL_MAX_CELLS_PER_STRATUM}" \
  --eval-gen-per-stratum "${EVAL_GEN_PER_STRATUM}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${extra[@]}"
