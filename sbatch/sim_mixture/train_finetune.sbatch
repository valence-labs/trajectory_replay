#!/bin/bash
#SBATCH --job-name=simmix-ft
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=sbatch/logs/sim_mixture_finetune_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Unconditional MoG finetune on task 1.
#
#   STRATEGY=trajectory LAMBDA_MIX=0.5 ./sbatch/sim_mixture/submit_finetune.sh
#   sbatch --export=ALL,STRATEGY=generative,LAMBDA_MIX=0.7 sbatch/sim_mixture/train_finetune.sbatch
#
# REPLAY_BACKEND=mean_flow additionally co-trains a dual-time MeanFlow with the
# two-branch identity loss (MF_LR, DIAG_FRACTION, TIME_MU, TIME_SIGMA, ADAPTIVE_P,
# MEAN_FLOW_STEPS). Defaults follow zhuyu CIFAR one-step schedule.

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${STRATEGY:=trajectory}"
: "${LAMBDA_MIX:=0.5}"
: "${RUN_NAME:=}"
: "${PRETRAIN_CHECKPOINT:=}"
: "${DATA_DIR:=}"
: "${EPOCHS:=100}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cuda}"
: "${ODE_METHOD:=euler}"
: "${ODE_STEP_SIZE:=0.05}"
: "${N_GEN_EVAL:=4000}"
: "${TRAJ_KAPPA:=none}"
: "${EVAL_EVERY:=0}"
: "${N_GEN_EVAL_CURVE:=}"
: "${REPLAY_BACKEND:=ode}"
: "${MEAN_FLOW_STEPS:=1}"
: "${MEAN_FLOW_CHECKPOINT:=}"
: "${MF_LR:=}"
: "${DIAG_FRACTION:=0.75}"
: "${TIME_MU:=2.0}"
: "${TIME_SIGMA:=2.0}"
: "${ADAPTIVE_P:=0}"
: "${MF_MAX_GRAD_NORM:=1.0}"
: "${MF_USE_ADAMW:=1}"
: "${MF_ADAM_BETA2:=0.95}"
: "${MF_WEIGHT_DECAY:=0.0}"

extra=()
if [[ -n "${RUN_NAME}" ]]; then
  extra+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${PRETRAIN_CHECKPOINT}" ]]; then
  extra+=(--pretrain-checkpoint "${PRETRAIN_CHECKPOINT}")
fi
if [[ -n "${DATA_DIR}" ]]; then
  extra+=(--data-dir "${DATA_DIR}")
fi
if [[ "${STRATEGY}" == "trajectory" ]]; then
  extra+=(--traj-kappa "${TRAJ_KAPPA}")
fi
if [[ "${EVAL_EVERY}" -gt 0 ]]; then
  extra+=(--eval-every "${EVAL_EVERY}")
fi
if [[ -n "${N_GEN_EVAL_CURVE}" ]]; then
  extra+=(--n-gen-eval-curve "${N_GEN_EVAL_CURVE}")
fi
if [[ "${REPLAY_BACKEND}" == "mean_flow" ]]; then
  extra+=(--replay-backend mean_flow)
  extra+=(--mean-flow-steps "${MEAN_FLOW_STEPS}")
  extra+=(--diag-fraction "${DIAG_FRACTION}")
  extra+=(--time-mu "${TIME_MU}")
  extra+=(--time-sigma "${TIME_SIGMA}")
  extra+=(--adaptive-p "${ADAPTIVE_P}")
  extra+=(--mf-max-grad-norm "${MF_MAX_GRAD_NORM}")
  extra+=(--mf-adam-beta2 "${MF_ADAM_BETA2}")
  extra+=(--mf-weight-decay "${MF_WEIGHT_DECAY}")
  if [[ "${MF_USE_ADAMW}" == "1" ]]; then
    extra+=(--mf-use-adamw)
  else
    extra+=(--mf-no-adamw)
  fi
  if [[ -n "${MF_LR}" ]]; then
    extra+=(--mf-lr "${MF_LR}")
  fi
  if [[ -n "${MEAN_FLOW_CHECKPOINT}" ]]; then
    extra+=(--mean-flow-checkpoint "${MEAN_FLOW_CHECKPOINT}")
  fi
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  STRATEGY=${STRATEGY}  LAMBDA_MIX=${LAMBDA_MIX}  DEVICE=${DEVICE}  TRAJ_KAPPA=${TRAJ_KAPPA}  EVAL_EVERY=${EVAL_EVERY}  REPLAY_BACKEND=${REPLAY_BACKEND}"
if [[ "${REPLAY_BACKEND}" == "mean_flow" ]]; then
  echo "MF_MAX_GRAD_NORM=${MF_MAX_GRAD_NORM}  MF_USE_ADAMW=${MF_USE_ADAMW}  MF_ADAM_BETA2=${MF_ADAM_BETA2}  MF_WEIGHT_DECAY=${MF_WEIGHT_DECAY}"
fi

uv sync
uv run python experiments/sim_mixture/train_finetune.py \
  --dim "${DIM}" \
  --strategy "${STRATEGY}" \
  --lambda-mix "${LAMBDA_MIX}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --ode-method "${ODE_METHOD}" \
  --ode-step-size "${ODE_STEP_SIZE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
