#!/bin/bash
#SBATCH --job-name=tx-cl
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=sbatch/logs/tx_cl_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1
#SBATCH --requeue

# Sequential ODE-CL in STATE SE (default), SCimilarity (REPRESENTATION=scimilarity),
# or gene expression (REPRESENTATION=genes). Embedding jobs load a sidecar into RAM.
# Gene jobs load h5ad X (Replogle HVGs ~5 GiB). Request 512G.
#
#   sbatch --export=ALL,DATASET=replogle,STRATEGY=generative,LAMBDA_MODE=grid \
#     sbatch/tx/train_cl.sbatch
#   sbatch --export=ALL,DATASET=pbmc,STRATEGY=trajectory,LAMBDA_MODE=equal \
#     sbatch/tx/train_cl.sbatch
#   SMOKE=1 for a short real-dump check.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results

: "${DATASET:=replogle}"
: "${STRATEGY:=generative}"
: "${REPRESENTATION:=se}"
: "${REPLAY_BACKEND:=ode}"
: "${LAMBDA_MODE:=grid}"
: "${MEAN_FLOW_STEPS:=1}"
: "${FINETUNE_SCOPE:=full}"
: "${FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS:=0}"
: "${FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS:=0}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"
: "${LORA_TRAIN_OUTPUT_LAYER:=0}"
: "${TWO_STAGE_1_EPOCHS:=0}"
: "${TWO_STAGE_1_EPOCH_FRACTION:=}"
: "${GFT_BETA_START:=1.0}"
: "${GFT_BETA_END:=0.0}"
: "${GFT_BETA_SCHEDULE:=sigmoid}"
: "${GFT_SIGMOID_MIDPOINT:=0.5}"
: "${GFT_SIGMOID_STEEPNESS:=10.0}"
: "${SEED:=42}"
: "${RUN_NAME:=}"
: "${DECODER_DIR:=}"
: "${REQUIRE_EXISTING_DECODERS:=0}"
: "${TARGET_STEPS:=100000}"
: "${LATER_TARGET_STEPS:=0}"
: "${LR:=1e-4}"
: "${LATER_LR:=0}"
: "${HIDDEN_DIM:=1024}"
: "${DEPTH:=4}"
: "${BACKBONE_TYPE:=mlp}"
: "${DEVICE:=cuda}"
: "${SMOKE:=0}"
: "${H5AD_PATH:=}"
: "${SE_PATH:=}"
: "${PCA_BASIS:=}"
: "${PCA_DIM:=}"
: "${NO_RESUME:=0}"
: "${RUN_HELDOUT_CURVE:=1}"
: "${RUN_CELLEVAL:=0}"
: "${PRED_CONTROL:=decoded}"
: "${FLOW_SOURCE:=noise}"
: "${EVAL_MAX_CELLS_PER_STRATUM:=0}"
: "${EVAL_MAX_CONTROL_CELLS:=0}"
: "${EVAL_GEN_PER_STRATUM:=500}"
: "${LAMBDA_GEN_PER_STRATUM:=100}"
: "${EVAL_FINAL_TRACKS:=fit+forget+stream+heldout}"
: "${REPLAY_CONDITION_SAMPLING:=proportional}"
: "${ODE_STEP_SIZE:=0.05}"
: "${TRAJECTORY_BATCH_SAMPLING:=equal}"
: "${TRAJECTORY_REPLAY_BATCH_MULTIPLIER:=1.0}"
: "${TRAJECTORY_REPLAY_LOSS_MULTIPLIER:=1.0}"
: "${TRAJECTORY_TIME_SAMPLING:=batch}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

extra=()
if [[ -n "${RUN_NAME}" ]]; then
  extra+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${DECODER_DIR}" ]]; then
  extra+=(--decoder-dir "${DECODER_DIR}")
fi
if [[ "${REQUIRE_EXISTING_DECODERS}" == "1" ]]; then
  extra+=(--require-existing-decoders)
fi
if [[ "${SMOKE}" == "1" ]]; then
  extra+=(--smoke)
fi
if [[ ! "${TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_STEPS must be a positive integer; got '${TARGET_STEPS}'" >&2
  exit 2
fi
echo "RUN_NAME=${RUN_NAME} TARGET_STEPS=${TARGET_STEPS} LATER_TARGET_STEPS=${LATER_TARGET_STEPS} REPRESENTATION=${REPRESENTATION} PCA_DIM=${PCA_DIM:-n/a} STRATEGY=${STRATEGY} REPLAY_BACKEND=${REPLAY_BACKEND} MEAN_FLOW_STEPS=${MEAN_FLOW_STEPS} LAMBDA_MODE=${LAMBDA_MODE} FINETUNE_SCOPE=${FINETUNE_SCOPE} SEED=${SEED} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
extra+=(--target-steps "${TARGET_STEPS}")
if [[ "${LATER_TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  extra+=(--later-target-steps "${LATER_TARGET_STEPS}")
elif [[ -n "${LATER_TARGET_STEPS}" && "${LATER_TARGET_STEPS}" != "0" ]]; then
  echo "LATER_TARGET_STEPS must be 0 or a positive integer; got '${LATER_TARGET_STEPS}'" >&2
  exit 2
fi
if [[ "${FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS}" == "1" ]]; then
  extra+=(--finetune-train-old-pert-embeddings)
fi
if [[ "${FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS}" == "1" ]]; then
  extra+=(--finetune-train-old-context-embeddings)
fi
if [[ "${LORA_TRAIN_OUTPUT_LAYER}" == "1" ]]; then
  extra+=(--lora-train-output-layer)
fi
if [[ -n "${TWO_STAGE_1_EPOCH_FRACTION}" ]]; then
  extra+=(--two-stage-1-epoch-fraction "${TWO_STAGE_1_EPOCH_FRACTION}")
fi
if [[ -n "${H5AD_PATH}" ]]; then
  extra+=(--h5ad-path "${H5AD_PATH}")
fi
if [[ -n "${SE_PATH}" ]]; then
  extra+=(--se-path "${SE_PATH}")
fi
if [[ -n "${PCA_BASIS}" ]]; then
  extra+=(--pca-basis "${PCA_BASIS}")
fi
if [[ "${REPRESENTATION}" == "pca" ]]; then
  # Keep in lockstep with DEFAULT_PCA_DIM in tx_preprocess.py.
  : "${PCA_DIM:=256}"
  extra+=(--pca-dim "${PCA_DIM}")
fi
# A fresh submission must not pick up an incompatible old checkpoint. If Slurm
# requeues this same job after preemption, however, resume its completed tasks
# instead of paying the full training budget again.
if [[ "${NO_RESUME}" == "1" && "${SLURM_RESTART_COUNT:-0}" == "0" ]]; then
  extra+=(--no-resume)
fi
if [[ "${PRED_CONTROL}" != "decoded" ]]; then
  extra+=(--pred-control "${PRED_CONTROL}")
fi
if [[ "${FLOW_SOURCE}" != "noise" ]]; then
  extra+=(--flow-source "${FLOW_SOURCE}")
fi
extra+=(--eval-max-cells-per-stratum "${EVAL_MAX_CELLS_PER_STRATUM}")
extra+=(--eval-max-control-cells "${EVAL_MAX_CONTROL_CELLS}")
extra+=(--eval-gen-per-stratum "${EVAL_GEN_PER_STRATUM}")
extra+=(--lambda-gen-per-stratum "${LAMBDA_GEN_PER_STRATUM}")
extra+=(--eval-final-tracks "${EVAL_FINAL_TRACKS}")
extra+=(--replay-condition-sampling "${REPLAY_CONDITION_SAMPLING}")
extra+=(--ode-step-size "${ODE_STEP_SIZE}")
extra+=(--later-lr "${LATER_LR}")
extra+=(--trajectory-batch-sampling "${TRAJECTORY_BATCH_SAMPLING}")
extra+=(--trajectory-replay-batch-multiplier "${TRAJECTORY_REPLAY_BATCH_MULTIPLIER}")
extra+=(--trajectory-replay-loss-multiplier "${TRAJECTORY_REPLAY_LOSS_MULTIPLIER}")
extra+=(--trajectory-time-sampling "${TRAJECTORY_TIME_SAMPLING}")

"${PYTHON_BIN}" experiments/tx/train_cl.py \
  --dataset "${DATASET}" \
  --representation "${REPRESENTATION}" \
  --strategy "${STRATEGY}" \
  --replay-backend "${REPLAY_BACKEND}" \
  --lambda-mode "${LAMBDA_MODE}" \
  --mean-flow-steps "${MEAN_FLOW_STEPS}" \
  --finetune-scope "${FINETUNE_SCOPE}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --two-stage-1-epochs "${TWO_STAGE_1_EPOCHS}" \
  --gft-beta-start "${GFT_BETA_START}" \
  --gft-beta-end "${GFT_BETA_END}" \
  --gft-beta-schedule "${GFT_BETA_SCHEDULE}" \
  --gft-sigmoid-midpoint "${GFT_SIGMOID_MIDPOINT}" \
  --gft-sigmoid-steepness "${GFT_SIGMOID_STEEPNESS}" \
  --lr "${LR}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --backbone-type "${BACKBONE_TYPE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${extra[@]}"

if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    echo "RUN_NAME or OUT_DIR is required to run post-CL eval in this job" >&2
    exit 1
  fi
  OUT_DIR="${REPO_ROOT}/results_dir/tx_results/${DATASET}/${RUN_NAME}"
fi

if [[ "${RUN_HELDOUT_CURVE}" == "1" ]]; then
  echo "=== Heldout Pearson-Δ curve (${OUT_DIR}) ===" >&2
  "${PYTHON_BIN}" experiments/tx/eval_cl_heldout.py \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}"
fi

if [[ "${RUN_CELLEVAL}" == "1" ]]; then
  echo "=== Cell-Eval minimal, final + heldout curve (${OUT_DIR}) ===" >&2
  "${PYTHON_BIN}" experiments/tx/eval_cl_celleval.py \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}"
  "${PYTHON_BIN}" experiments/tx/eval_cl_celleval.py \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --heldout-curve
fi
