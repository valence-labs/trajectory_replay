#!/bin/bash
#SBATCH --job-name=simmix-distill-mix
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/sim_mixture_distill_mix_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Distill MeanFlow directly on the mixture data from a frozen oracle CFM
# (no CL, no replay) to disentangle N=1 quality: replay artifact vs inherent
# MeanFlow limitation. Uses the zhuyu CIFAR one-step schedule by default.
#
#   LAMBDA_MIX=0.1 ./sbatch/sim_mixture/submit_distill_meanflow_mixture.sh

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${LAMBDA_MIX:=0.5}"
: "${ORACLE_CHECKPOINT:=}"
: "${DISTILL_EPOCHS:=400}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${TRAIN_SEED:=0}"
: "${DIAG_FRACTION:=0.25}"
: "${CORNER_FRACTION:=0.0}"
: "${TIME_MU:=-2.0}"
: "${TIME_SIGMA:=2.0}"
: "${ADAPTIVE_P:=0.0}"
: "${DEVICE:=cuda}"
: "${N_GEN_EVAL:=4000}"
: "${NFE_GRID:=1,2,4,8}"
: "${RUN_NAME:=}"
: "${USE_ADAMW:=1}"
: "${ADAM_BETA2:=0.95}"
: "${WEIGHT_DECAY:=0.0}"
: "${MAX_GRAD_NORM:=1.0}"

extra=()
if [[ -n "${ORACLE_CHECKPOINT}" ]]; then
  extra+=(--oracle-checkpoint "${ORACLE_CHECKPOINT}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  extra+=(--run-name "${RUN_NAME}")
fi
if [[ "${USE_ADAMW}" == "1" ]]; then
  extra+=(--use-adamw)
else
  extra+=(--no-adamw)
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "Distill-on-mixture λ=${LAMBDA_MIX}  DIM=${DIM}  DEVICE=${DEVICE}"
echo "USE_ADAMW=${USE_ADAMW} ADAM_BETA2=${ADAM_BETA2} WEIGHT_DECAY=${WEIGHT_DECAY} MAX_GRAD_NORM=${MAX_GRAD_NORM}"

uv sync
uv run python experiments/sim_mixture/train_distill_meanflow_mixture.py \
  --dim "${DIM}" \
  --lambda-mix "${LAMBDA_MIX}" \
  --distill-epochs "${DISTILL_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --diag-fraction "${DIAG_FRACTION}" \
  --corner-fraction "${CORNER_FRACTION}" \
  --time-mu "${TIME_MU}" \
  --time-sigma "${TIME_SIGMA}" \
  --adaptive-p "${ADAPTIVE_P}" \
  --device "${DEVICE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  --nfe-grid "${NFE_GRID}" \
  --adam-beta2 "${ADAM_BETA2}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
