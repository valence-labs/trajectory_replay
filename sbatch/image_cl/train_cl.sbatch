#!/bin/bash
#SBATCH --job-name=image-cl
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/image_cl_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Split-FMNIST unconditional CFM continual learning (Masip UNet backbone).
#
#   sbatch --export=ALL,STRATEGY=trajectory,SEED=42 sbatch/image_cl/train_cl.sbatch
#   sbatch --export=ALL,STRATEGY=trajectory,SEED=42,REPLAY_BACKEND=mean_flow \
#     sbatch/image_cl/train_cl.sbatch
#
# Env overrides: STRATEGY, SEED, EPOCHS, BATCH_SIZE, LR, LAMBDA_GRID, LAMBDA_MODE,
# TRAJ_KAPPA, REPLAY_BACKEND (ode|mean_flow), MEAN_FLOW_STEPS, MEAN_FLOW_LR,
# DISTILL_EPOCHS, DISTILL_LR, DISTILL_BATCH_SIZE,
# FID_VAL_SAMPLES, FID_TEST_SAMPLES, KLD_CLF_PATH, NO_KLD, RUN_NAME, SMOKE,
# CLASS_CONDITIONAL (0|1), NUM_CLASSES, LABEL_DROPOUT, GUIDANCE_SCALE,
# UNET_SIZE, MAX_TASKS, N_EXPERIENCES, PER_CLASS_FID_SAMPLES, MAX_GRAD_NORM,
# DISABLE_FUSED_SDPA, QOS via sbatch.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

: "${STRATEGY:=trajectory}"
: "${SEED:=42}"
: "${DATASET:=split_fmnist}"
case "${DATASET}" in
  split_cifar10)
    : "${RESULTS_SLUG:=cifar10}"
    : "${EPOCHS:=200}"
    : "${BATCH_SIZE:=64}"
    : "${ODE_STEP_SIZE:=0.05}"
    : "${DISTILL_EPOCHS:=200}"
    : "${N_EXPERIENCES:=5}"
    # zhuyu CIFAR schedule mirrored from data→noise to our noise→data time.
    : "${MEAN_FLOW_DIAG_FRACTION:=0.25}"
    : "${MEAN_FLOW_TIME_MU:=2}"
    : "${MEAN_FLOW_TIME_SIGMA:=2}"
    : "${MEAN_FLOW_ADAPTIVE_P:=0.75}"
    ;;
  split_imagenet100)
    : "${RESULTS_SLUG:=imagenet100}"
    : "${EPOCHS:=667}"
    : "${BATCH_SIZE:=256}"
    : "${ODE_STEP_SIZE:=0.02}"
    : "${DISTILL_EPOCHS:=150}"
    : "${N_EXPERIENCES:=3}"
    : "${LR:=1e-4}"
    : "${CLASS_CONDITIONAL:=1}"
    : "${UNET_SIZE:=large}"
    : "${MAX_GRAD_NORM:=1.0}"
    : "${PER_CLASS_FID_SAMPLES:=0}"
    : "${FID_TEST_SAMPLES:=10000}"
    : "${FID_VAL_SAMPLES:=2048}"
    : "${NO_KLD:=1}"
    : "${CUMULATIVE_SCHEDULE:=steps}"
    # zhuyu ImageNet schedule mirrored from data→noise to our noise→data time.
    : "${MEAN_FLOW_DIAG_FRACTION:=0.75}"
    : "${MEAN_FLOW_TIME_MU:=0.4}"
    : "${MEAN_FLOW_TIME_SIGMA:=1.0}"
    : "${MEAN_FLOW_ADAPTIVE_P:=1.0}"
    ;;
  *)
    : "${RESULTS_SLUG:=fmnist}"
    : "${EPOCHS:=100}"
    : "${BATCH_SIZE:=128}"
    : "${ODE_STEP_SIZE:=0.1}"
    : "${DISTILL_EPOCHS:=100}"
    : "${N_EXPERIENCES:=5}"
    # zhuyu CIFAR schedule (also FMNIST), mirrored to noise→data time.
    : "${MEAN_FLOW_DIAG_FRACTION:=0.25}"
    : "${MEAN_FLOW_TIME_MU:=2}"
    : "${MEAN_FLOW_TIME_SIGMA:=2}"
    : "${MEAN_FLOW_ADAPTIVE_P:=0.75}"
    ;;
esac
# Optional override: Euler step for ODE teacher replay only (eval keeps ODE_STEP_SIZE).
: "${REPLAY_ODE_STEP_SIZE:=}"
: "${CUMULATIVE_SCHEDULE:=epochs}"
: "${LR:=2e-4}"
: "${LAMBDA_GRID:=0.1 0.25 0.5 0.75 0.9}"
: "${TRAJ_KAPPA:=none}"
: "${LAMBDA_MODE:=grid}"
: "${REPLAY_BACKEND:=ode}"
: "${MEAN_FLOW_STEPS:=1}"
: "${MEAN_FLOW_ADAPTIVE_P:=0.75}"
: "${MEAN_FLOW_TIME_SAMPLER:=logit_normal}"
: "${MEAN_FLOW_LR:=1e-4}"
: "${MEAN_FLOW_MAX_GRAD_NORM:=1.0}"
: "${MEAN_FLOW_USE_ADAMW:=1}"
: "${MEAN_FLOW_ADAM_BETA2:=0.95}"
: "${MEAN_FLOW_WEIGHT_DECAY:=0.0}"
# Task-0 distill shares MeanFlow optimizer hparams (not CFM --lr).
: "${DISTILL_LR:=${MEAN_FLOW_LR}}"
: "${DISTILL_BATCH_SIZE:=${BATCH_SIZE}}"
: "${MEAN_FLOW_HIDDEN_DIM:=512}"
: "${MEAN_FLOW_DEPTH:=6}"
: "${ODE_METHOD:=euler}"
: "${FID_VAL_SAMPLES:=2048}"
: "${FID_TEST_SAMPLES:=10000}"
: "${FID_BATCH_SIZE:=128}"
: "${IMAGE_SIZE:=32}"
: "${VAL_FRACTION:=0.1}"
: "${DEVICE:=cuda}"
: "${SMOKE:=0}"
: "${NO_KLD:=0}"
: "${DISABLE_FUSED_SDPA:=1}"
: "${CLASS_CONDITIONAL:=0}"
: "${LABEL_DROPOUT:=0.1}"
: "${GUIDANCE_SCALE:=1.5}"
: "${UNET_SIZE:=medium}"
: "${MAX_TASKS:=}"
: "${PER_CLASS_FID_SAMPLES:=0}"
: "${MAX_GRAD_NORM:=0}"
# Memory-free finetuning controls (STRATEGY=lora|gft). LORA_ALPHA defaults to 2r.
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=}"
: "${LORA_TRAIN_OUTPUT_LAYER:=0}"
: "${GFT_BETA_START:=1.0}"
: "${GFT_BETA_END:=0.0}"
: "${GFT_BETA_SCHEDULE:=sigmoid}"
# Reuse a completed task 0 from another run (e.g. image_cl/fmnist/naive_s42).
# Task 0 is identical across strategies at a given seed, so this is exact.
: "${TASK0_FROM:=}"
# Prefer an explicit RUN_NAME from the submit script. Fallback naming mirrors
# CIFAR/FMNIST for the locked ImageNet recipe (no _cc/_large/_clip/_e3 tags).
if [[ -z "${RUN_NAME:-}" ]]; then
  stem="${STRATEGY}"
  if [[ "${STRATEGY}" == "joint" ]]; then
    stem="cumulative"
  fi
  suffix=""
  if [[ "${DATASET}" == "split_imagenet100" ]]; then
    if [[ "${CLASS_CONDITIONAL}" != "1" ]]; then
      suffix="${suffix}_uncond"
    fi
    if [[ "${UNET_SIZE}" != "large" ]]; then
      suffix="${suffix}_${UNET_SIZE}"
    fi
    if [[ "${MAX_GRAD_NORM}" == "0" || "${MAX_GRAD_NORM}" == "0.0" ]]; then
      suffix="${suffix}_noclip"
    fi
    if [[ "${N_EXPERIENCES}" != "3" ]]; then
      suffix="${suffix}_e${N_EXPERIENCES}"
    fi
  else
    if [[ "${CLASS_CONDITIONAL}" == "1" ]]; then
      suffix="${suffix}_cc"
    fi
    if [[ "${UNET_SIZE}" == "large" ]]; then
      suffix="${suffix}_large"
    fi
    if [[ "${MAX_GRAD_NORM}" != "0" && "${MAX_GRAD_NORM}" != "0.0" ]]; then
      suffix="${suffix}_clip"
    fi
    if [[ "${N_EXPERIENCES}" != "5" ]]; then
      suffix="${suffix}_e${N_EXPERIENCES}"
    fi
  fi
  if [[ "${REPLAY_BACKEND}" == "mean_flow" && ( "${STRATEGY}" == "generative" || "${STRATEGY}" == "trajectory" ) ]]; then
    suffix="${suffix}_mf_n${MEAN_FLOW_STEPS}"
  fi
  if [[ "${LAMBDA_MODE}" == "equal" && ( "${STRATEGY}" == "generative" || "${STRATEGY}" == "trajectory" ) ]]; then
    suffix="${suffix}_eq"
  fi
  if [[ "${STRATEGY}" == "lora" ]]; then
    stem="lora_r${LORA_RANK}"
  fi
  if [[ "${STRATEGY}" == "gft" ]]; then
    stem="gft_${GFT_BETA_SCHEDULE}"
  fi
  if [[ -n "${MAX_TASKS}" ]]; then
    suffix="${suffix}_t${MAX_TASKS}"
  fi
  RUN_NAME="image_cl/${RESULTS_SLUG}/${stem}${suffix}_s${SEED}"
fi

export DISABLE_FUSED_SDPA

extra=()
if [[ "${SMOKE}" == "1" ]]; then
  extra+=(--smoke)
fi
if [[ "${NO_KLD}" == "1" ]]; then
  extra+=(--no-kld)
fi
if [[ "${CLASS_CONDITIONAL}" == "1" ]]; then
  extra+=(--class-conditional)
  extra+=(--label-dropout "${LABEL_DROPOUT}")
  extra+=(--guidance-scale "${GUIDANCE_SCALE}")
  if [[ -n "${NUM_CLASSES:-}" ]]; then
    extra+=(--num-classes "${NUM_CLASSES}")
  fi
fi
if [[ "${STRATEGY}" == "lora" ]]; then
  extra+=(--lora-rank "${LORA_RANK}")
  extra+=(--lora-alpha "${LORA_ALPHA:-$(awk "BEGIN{print 2*${LORA_RANK}}")}")
  if [[ "${LORA_TRAIN_OUTPUT_LAYER}" == "1" ]]; then
    extra+=(--lora-train-output-layer)
  fi
fi
if [[ "${STRATEGY}" == "gft" ]]; then
  extra+=(--gft-beta-start "${GFT_BETA_START}")
  extra+=(--gft-beta-end "${GFT_BETA_END}")
  extra+=(--gft-beta-schedule "${GFT_BETA_SCHEDULE}")
fi
extra+=(--unet-size "${UNET_SIZE}")
extra+=(--max-grad-norm "${MAX_GRAD_NORM}")
extra+=(--n-experiences "${N_EXPERIENCES}")
if [[ -n "${MAX_TASKS}" ]]; then
  extra+=(--max-tasks "${MAX_TASKS}")
fi
if [[ "${PER_CLASS_FID_SAMPLES}" != "0" ]]; then
  extra+=(--per-class-fid-samples "${PER_CLASS_FID_SAMPLES}")
fi
if [[ -n "${KLD_CLF_PATH:-}" ]]; then
  extra+=(--kld-clf-path "${KLD_CLF_PATH}")
fi
if [[ -n "${LATENT_DIR:-}" ]]; then
  extra+=(--latent-dir "${LATENT_DIR}")
fi

# shellcheck disable=SC2206
lam_arr=(${LAMBDA_GRID})

echo "Job ${SLURM_JOB_ID:-local} image_cl at $(date -Is)"
echo "  DATASET=${DATASET} STRATEGY=${STRATEGY} SEED=${SEED} RUN_NAME=${RUN_NAME}"
echo "  EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} LR=${LR} ODE_STEP_SIZE=${ODE_STEP_SIZE}"
echo "  REPLAY_ODE_STEP_SIZE=${REPLAY_ODE_STEP_SIZE:-<same as ODE_STEP_SIZE>}"
echo "  CUMULATIVE_SCHEDULE=${CUMULATIVE_SCHEDULE}"
echo "  LAMBDA_MODE=${LAMBDA_MODE} LAMBDA_GRID=${LAMBDA_GRID} TRAJ_KAPPA=${TRAJ_KAPPA}"
echo "  REPLAY_BACKEND=${REPLAY_BACKEND} MEAN_FLOW_STEPS=${MEAN_FLOW_STEPS}"
echo "  MEAN_FLOW_DIAG_FRACTION=${MEAN_FLOW_DIAG_FRACTION} MEAN_FLOW_ADAPTIVE_P=${MEAN_FLOW_ADAPTIVE_P}"
echo "  MEAN_FLOW_TIME_MU=${MEAN_FLOW_TIME_MU} MEAN_FLOW_TIME_SIGMA=${MEAN_FLOW_TIME_SIGMA}"
echo "  MEAN_FLOW_ADAM_BETA2=${MEAN_FLOW_ADAM_BETA2} MEAN_FLOW_MAX_GRAD_NORM=${MEAN_FLOW_MAX_GRAD_NORM}"
echo "  DISTILL_EPOCHS=${DISTILL_EPOCHS} NO_KLD=${NO_KLD}"
echo "  CLASS_CONDITIONAL=${CLASS_CONDITIONAL} LABEL_DROPOUT=${LABEL_DROPOUT} GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "  UNET_SIZE=${UNET_SIZE} N_EXPERIENCES=${N_EXPERIENCES} MAX_TASKS=${MAX_TASKS:-all} PER_CLASS_FID_SAMPLES=${PER_CLASS_FID_SAMPLES}"
echo "  MAX_GRAD_NORM=${MAX_GRAD_NORM}"
echo "  DISABLE_FUSED_SDPA=${DISABLE_FUSED_SDPA}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

# Drop Cursor/login-node UV cache paths leaked via --export=ALL (not writable on compute).
unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

if [[ -n "${REPLAY_ODE_STEP_SIZE}" ]]; then
  extra+=(--replay-ode-step-size "${REPLAY_ODE_STEP_SIZE}")
fi
extra+=(--mean-flow-time-mu "${MEAN_FLOW_TIME_MU}")
extra+=(--mean-flow-time-sigma "${MEAN_FLOW_TIME_SIGMA}")
if [[ -n "${MEAN_FLOW_LR}" ]]; then
  extra+=(--mean-flow-lr "${MEAN_FLOW_LR}")
fi

# Seed task 0 from a completed run so the trainer resumes at task 1. Only copy
# into a run that has not started task 0 itself, and never overwrite.
if [[ -n "${TASK0_FROM}" ]]; then
  src="${REPO_ROOT}/results_dir/${TASK0_FROM}/tasks/00"
  dst="${REPO_ROOT}/results_dir/${RUN_NAME}/tasks/00"
  if [[ ! -f "${src}/checkpoint.pt" || ! -f "${src}/metrics.json" ]]; then
    echo "TASK0_FROM=${TASK0_FROM} has no completed task 0 at ${src}" >&2
    exit 2
  fi
  if [[ -f "${dst}/metrics.json" ]]; then
    echo "  task 0 already present at ${dst}; leaving it alone"
  else
    mkdir -p "${dst}"
    cp -n "${src}"/* "${dst}/"
    echo "  seeded task 0 from ${TASK0_FROM}"
  fi
fi

# Use the prebuilt venv rather than `uv sync`: rebuilding the project on every
# job re-resolves build-system.requires against the internal mirror, which has
# intermittently 404'd on setuptools and left jobs hanging in the build step.
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "No interpreter at ${PYTHON_BIN}; run 'uv sync' once on a login node" >&2
  exit 2
fi
# train_cl.py imports `experiments.*` and `trajectory_replay.*` without
# touching sys.path, so both roots have to be on PYTHONPATH here.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" experiments/image_cl/train_cl.py \
  --dataset "${DATASET}" \
  --strategy "${STRATEGY}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --cumulative-schedule "${CUMULATIVE_SCHEDULE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lambda-grid "${lam_arr[@]}" \
  --lambda-mode "${LAMBDA_MODE}" \
  --traj-kappa "${TRAJ_KAPPA}" \
  --replay-backend "${REPLAY_BACKEND}" \
  --mean-flow-steps "${MEAN_FLOW_STEPS}" \
  --mean-flow-diag-fraction "${MEAN_FLOW_DIAG_FRACTION}" \
  --mean-flow-adaptive-p "${MEAN_FLOW_ADAPTIVE_P}" \
  --mean-flow-time-sampler "${MEAN_FLOW_TIME_SAMPLER}" \
  --mean-flow-max-grad-norm "${MEAN_FLOW_MAX_GRAD_NORM}" \
  --mean-flow-use-adamw "${MEAN_FLOW_USE_ADAMW}" \
  --mean-flow-adam-beta2 "${MEAN_FLOW_ADAM_BETA2}" \
  --mean-flow-weight-decay "${MEAN_FLOW_WEIGHT_DECAY}" \
  --distill-epochs "${DISTILL_EPOCHS}" \
  --distill-lr "${DISTILL_LR}" \
  --distill-batch-size "${DISTILL_BATCH_SIZE}" \
  --mean-flow-hidden-dim "${MEAN_FLOW_HIDDEN_DIM}" \
  --mean-flow-depth "${MEAN_FLOW_DEPTH}" \
  --ode-method "${ODE_METHOD}" \
  --ode-step-size "${ODE_STEP_SIZE}" \
  --fid-val-samples "${FID_VAL_SAMPLES}" \
  --fid-test-samples "${FID_TEST_SAMPLES}" \
  --fid-batch-size "${FID_BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --val-fraction "${VAL_FRACTION}" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
echo "Results: ${REPO_ROOT}/results_dir/${RUN_NAME}"
