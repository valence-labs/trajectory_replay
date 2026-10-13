#!/usr/bin/env bash
# Submit Split-FMNIST image_cl grid: strategies × seeds [× replay backends] [× λ modes].
#
#   ./sbatch/image_cl/submit_fmnist_min.sh
#   STRATEGIES="trajectory" SEEDS="42" ./sbatch/image_cl/submit_fmnist_min.sh
#   # Fixed equal-weight λ=(n-1)/n (no sweep):
#   STRATEGIES="trajectory" LAMBDA_MODE=equal SEEDS="42 69 1714" \
#     ./sbatch/image_cl/submit_fmnist_min.sh
#   # ODE vs MeanFlow:
#   STRATEGIES="trajectory" REPLAY_BACKENDS="ode mean_flow" SEEDS="42 69 1714" \
#     ./sbatch/image_cl/submit_fmnist_min.sh
#   # Full FMNIST MeanFlow table (N=1 and N=4):
#   ./sbatch/image_cl/submit_fmnist_meanflow.sh
#   DRY_RUN=1 ./sbatch/image_cl/submit_fmnist_min.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH="${REPO_ROOT}/sbatch/image_cl/train_cl.sbatch"

STRATEGIES=(${STRATEGIES:-naive generative trajectory})
SEEDS=(${SEEDS:-42 69 1714})
REPLAY_BACKENDS=(${REPLAY_BACKENDS:-ode})
LAMBDA_MODE="${LAMBDA_MODE:-grid}"
LAMBDA_GRID="${LAMBDA_GRID:-0.1 0.25 0.5 0.75 0.9}"
# Optional run-name tag, e.g. RUN_SUFFIX=g3 → generative_g3_s42 (avoids clobbering other grids).
RUN_SUFFIX="${RUN_SUFFIX:-}"
# Optional walltime override for sbatch, e.g. TIME=30:00:00
TIME="${TIME:-}"
QOS="${QOS:-}"  # optional override, e.g. QOS=hooke-predict
DRY_RUN="${DRY_RUN:-0}"
DISABLE_FUSED_SDPA="${DISABLE_FUSED_SDPA:-1}"
TRAJ_KAPPA="${TRAJ_KAPPA:-none}"
DATASET="${DATASET:-split_fmnist}"
case "${DATASET}" in
  split_cifar10)
    RESULTS_SLUG="${RESULTS_SLUG:-cifar10}"
    EPOCHS="${EPOCHS:-200}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.05}"
    DISTILL_EPOCHS="${DISTILL_EPOCHS:-200}"
    N_EXPERIENCES="${N_EXPERIENCES:-5}"
    MEAN_FLOW_DIAG_FRACTION="${MEAN_FLOW_DIAG_FRACTION:-0.25}"
    MEAN_FLOW_TIME_MU="${MEAN_FLOW_TIME_MU:-2}"
    MEAN_FLOW_TIME_SIGMA="${MEAN_FLOW_TIME_SIGMA:-2}"
    MEAN_FLOW_ADAPTIVE_P="${MEAN_FLOW_ADAPTIVE_P:-0.75}"
    ;;
  split_imagenet100)
    RESULTS_SLUG="${RESULTS_SLUG:-imagenet100}"
    # Locked class-cond recipe (~100k steps/task @ bs 256).
    EPOCHS="${EPOCHS:-667}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.02}"
    DISTILL_EPOCHS="${DISTILL_EPOCHS:-150}"
    N_EXPERIENCES="${N_EXPERIENCES:-3}"
    CLASS_CONDITIONAL="${CLASS_CONDITIONAL:-1}"
    UNET_SIZE="${UNET_SIZE:-large}"
    MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
    LR="${LR:-1e-4}"
    PER_CLASS_FID_SAMPLES="${PER_CLASS_FID_SAMPLES:-0}"
    FID_TEST_SAMPLES="${FID_TEST_SAMPLES:-10000}"
    FID_VAL_SAMPLES="${FID_VAL_SAMPLES:-2048}"
    NO_KLD="${NO_KLD:-1}"
    CUMULATIVE_SCHEDULE="${CUMULATIVE_SCHEDULE:-steps}"
    MEAN_FLOW_DIAG_FRACTION="${MEAN_FLOW_DIAG_FRACTION:-0.75}"
    MEAN_FLOW_TIME_MU="${MEAN_FLOW_TIME_MU:-0.4}"
    MEAN_FLOW_TIME_SIGMA="${MEAN_FLOW_TIME_SIGMA:-1.0}"
    MEAN_FLOW_ADAPTIVE_P="${MEAN_FLOW_ADAPTIVE_P:-1.0}"
    ;;
  *)
    RESULTS_SLUG="${RESULTS_SLUG:-fmnist}"
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.1}"
    DISTILL_EPOCHS="${DISTILL_EPOCHS:-100}"
    N_EXPERIENCES="${N_EXPERIENCES:-5}"
    MEAN_FLOW_DIAG_FRACTION="${MEAN_FLOW_DIAG_FRACTION:-0.25}"
    MEAN_FLOW_TIME_MU="${MEAN_FLOW_TIME_MU:-2}"
    MEAN_FLOW_TIME_SIGMA="${MEAN_FLOW_TIME_SIGMA:-2}"
    MEAN_FLOW_ADAPTIVE_P="${MEAN_FLOW_ADAPTIVE_P:-0.75}"
    ;;
esac
MEAN_FLOW_STEPS="${MEAN_FLOW_STEPS:-1}"
MEAN_FLOW_ADAPTIVE_P="${MEAN_FLOW_ADAPTIVE_P:-0.75}"
MEAN_FLOW_LR="${MEAN_FLOW_LR:-1e-4}"
MEAN_FLOW_MAX_GRAD_NORM="${MEAN_FLOW_MAX_GRAD_NORM:-1.0}"
MEAN_FLOW_USE_ADAMW="${MEAN_FLOW_USE_ADAMW:-1}"
MEAN_FLOW_ADAM_BETA2="${MEAN_FLOW_ADAM_BETA2:-0.95}"
DISTILL_LR="${DISTILL_LR:-${MEAN_FLOW_LR}}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-${BATCH_SIZE}}"
CLASS_CONDITIONAL="${CLASS_CONDITIONAL:-0}"
LABEL_DROPOUT="${LABEL_DROPOUT:-0.1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
UNET_SIZE="${UNET_SIZE:-medium}"
MAX_TASKS="${MAX_TASKS:-}"
PER_CLASS_FID_SAMPLES="${PER_CLASS_FID_SAMPLES:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0}"
CUMULATIVE_SCHEDULE="${CUMULATIVE_SCHEDULE:-epochs}"
LR="${LR:-}"
FID_TEST_SAMPLES="${FID_TEST_SAMPLES:-}"
FID_VAL_SAMPLES="${FID_VAL_SAMPLES:-}"
NO_KLD="${NO_KLD:-0}"

# ImageNet locked-recipe defaults (suffixes only for deviations → CIFAR-like names).
IMAGENET_DEFAULT_CC=0
IMAGENET_DEFAULT_UNET=""
IMAGENET_DEFAULT_CLIP=""
IMAGENET_DEFAULT_NEXP=""
if [[ "${DATASET}" == "split_imagenet100" ]]; then
  IMAGENET_DEFAULT_CC=1
  IMAGENET_DEFAULT_UNET="large"
  IMAGENET_DEFAULT_CLIP="1.0"
  IMAGENET_DEFAULT_NEXP="3"
fi

echo "STRATEGIES=${STRATEGIES[*]}"
echo "SEEDS=${SEEDS[*]} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} ODE_STEP_SIZE=${ODE_STEP_SIZE} TRAJ_KAPPA=${TRAJ_KAPPA}"
echo "REPLAY_ODE_STEP_SIZE=${REPLAY_ODE_STEP_SIZE:-<same as ODE_STEP_SIZE>}"
echo "REPLAY_BACKENDS=${REPLAY_BACKENDS[*]} LAMBDA_MODE=${LAMBDA_MODE} LAMBDA_GRID=${LAMBDA_GRID} QOS=${QOS:-default}"
echo "DISTILL_EPOCHS=${DISTILL_EPOCHS} DISABLE_FUSED_SDPA=${DISABLE_FUSED_SDPA}"
echo "MEAN_FLOW_STEPS=${MEAN_FLOW_STEPS} MEAN_FLOW_LR=${MEAN_FLOW_LR} DISTILL_LR=${DISTILL_LR} DISTILL_BATCH_SIZE=${DISTILL_BATCH_SIZE}"
echo "DIAG=${MEAN_FLOW_DIAG_FRACTION} ADAPTIVE_P=${MEAN_FLOW_ADAPTIVE_P} μ=${MEAN_FLOW_TIME_MU} σ=${MEAN_FLOW_TIME_SIGMA}"
echo "CLASS_CONDITIONAL=${CLASS_CONDITIONAL} LABEL_DROPOUT=${LABEL_DROPOUT} GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "UNET_SIZE=${UNET_SIZE} MAX_GRAD_NORM=${MAX_GRAD_NORM} LR=${LR:-default}"
echo "N_EXPERIENCES=${N_EXPERIENCES} MAX_TASKS=${MAX_TASKS:-all} PER_CLASS_FID_SAMPLES=${PER_CLASS_FID_SAMPLES}"
echo "FID_TEST_SAMPLES=${FID_TEST_SAMPLES:-sbatch-default} FID_VAL_SAMPLES=${FID_VAL_SAMPLES:-sbatch-default} NO_KLD=${NO_KLD}"
echo "CUMULATIVE_SCHEDULE=${CUMULATIVE_SCHEDULE}"
echo "DATASET=${DATASET} RESULTS_SLUG=${RESULTS_SLUG} RUN_SUFFIX=${RUN_SUFFIX:-none} TIME=${TIME:-sbatch-default}"

n=0
for strategy in "${STRATEGIES[@]}"; do
  for backend in "${REPLAY_BACKENDS[@]}"; do
    # Naive / cumulative / joint never use a replay backend; skip duplicate MF jobs.
    if [[ "${strategy}" == "naive" || "${strategy}" == "cumulative" || "${strategy}" == "joint" ]]; then
      if [[ "${backend}" != "ode" ]]; then
        continue
      fi
    fi
    for seed in "${SEEDS[@]}"; do
      suffix=""
      job_extra=""
      # Class-cond tag: always for non-ImageNet; on ImageNet only if *not* the default.
      if [[ "${DATASET}" == "split_imagenet100" ]]; then
        if [[ "${CLASS_CONDITIONAL}" != "${IMAGENET_DEFAULT_CC}" ]]; then
          if [[ "${CLASS_CONDITIONAL}" == "1" ]]; then
            suffix="${suffix}_cc"
            job_extra="${job_extra}-cc"
          else
            suffix="${suffix}_uncond"
            job_extra="${job_extra}-uncond"
          fi
        fi
        if [[ "${UNET_SIZE}" != "${IMAGENET_DEFAULT_UNET}" ]]; then
          suffix="${suffix}_${UNET_SIZE}"
          job_extra="${job_extra}-${UNET_SIZE:0:2}"
        fi
        if [[ "${MAX_GRAD_NORM}" != "${IMAGENET_DEFAULT_CLIP}" && "${MAX_GRAD_NORM}" != "0" && "${MAX_GRAD_NORM}" != "0.0" ]]; then
          suffix="${suffix}_clip"
          job_extra="${job_extra}-clip"
        elif [[ "${MAX_GRAD_NORM}" == "0" || "${MAX_GRAD_NORM}" == "0.0" ]]; then
          if [[ -n "${IMAGENET_DEFAULT_CLIP}" ]]; then
            suffix="${suffix}_noclip"
            job_extra="${job_extra}-noclip"
          fi
        fi
        if [[ "${N_EXPERIENCES}" != "${IMAGENET_DEFAULT_NEXP}" ]]; then
          suffix="${suffix}_e${N_EXPERIENCES}"
          job_extra="${job_extra}-e${N_EXPERIENCES}"
        fi
      else
        if [[ "${CLASS_CONDITIONAL}" == "1" ]]; then
          suffix="${suffix}_cc"
          job_extra="${job_extra}-cc"
        fi
        if [[ "${UNET_SIZE}" == "large" ]]; then
          suffix="${suffix}_large"
          job_extra="${job_extra}-lg"
        fi
        if [[ "${MAX_GRAD_NORM}" != "0" && "${MAX_GRAD_NORM}" != "0.0" ]]; then
          suffix="${suffix}_clip"
          job_extra="${job_extra}-clip"
        fi
        if [[ "${N_EXPERIENCES}" != "5" ]]; then
          suffix="${suffix}_e${N_EXPERIENCES}"
          job_extra="${job_extra}-e${N_EXPERIENCES}"
        fi
      fi
      if [[ "${backend}" == "mean_flow" ]]; then
        suffix="${suffix}_mf_n${MEAN_FLOW_STEPS}"
        job_extra="${job_extra}-mf-n${MEAN_FLOW_STEPS}"
      fi
      # λ-equal / RUN_SUFFIX only for strategies that use λ.
      if [[ "${strategy}" == "generative" || "${strategy}" == "trajectory" ]]; then
        if [[ "${LAMBDA_MODE}" == "equal" ]]; then
          suffix="${suffix}_eq"
          job_extra="${job_extra}-eq"
        fi
        if [[ -n "${RUN_SUFFIX}" ]]; then
          suffix="${suffix}_${RUN_SUFFIX}"
          job_extra="${job_extra}-${RUN_SUFFIX}"
        fi
      fi
      if [[ -n "${MAX_TASKS}" ]]; then
        suffix="${suffix}_t${MAX_TASKS}"
        job_extra="${job_extra}-t${MAX_TASKS}"
      fi
      # cumulative/joint share one result stem (Masip joint upper bound).
      stem="${strategy}"
      if [[ "${strategy}" == "joint" ]]; then
        stem="cumulative"
      fi
      run_name="image_cl/${RESULTS_SLUG}/${stem}${suffix}_s${seed}"
      job="icl-${stem:0:4}${job_extra}-s${seed}"
      export_vars="ALL,STRATEGY=${strategy},SEED=${seed},RUN_NAME=${run_name}"
      export_vars+=",DATASET=${DATASET},RESULTS_SLUG=${RESULTS_SLUG}"
      export_vars+=",REPO_ROOT=${REPO_ROOT},DISABLE_FUSED_SDPA=${DISABLE_FUSED_SDPA}"
      export_vars+=",TRAJ_KAPPA=${TRAJ_KAPPA},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE}"
      export_vars+=",ODE_STEP_SIZE=${ODE_STEP_SIZE}"
      if [[ -n "${REPLAY_ODE_STEP_SIZE:-}" ]]; then
        export_vars+=",REPLAY_ODE_STEP_SIZE=${REPLAY_ODE_STEP_SIZE}"
      fi
      export_vars+=",CUMULATIVE_SCHEDULE=${CUMULATIVE_SCHEDULE}"
      export_vars+=",REPLAY_BACKEND=${backend},DISTILL_EPOCHS=${DISTILL_EPOCHS}"
      export_vars+=",MEAN_FLOW_STEPS=${MEAN_FLOW_STEPS},LAMBDA_MODE=${LAMBDA_MODE}"
      export_vars+=",LAMBDA_GRID=${LAMBDA_GRID}"
      export_vars+=",MEAN_FLOW_LR=${MEAN_FLOW_LR},DISTILL_LR=${DISTILL_LR}"
      export_vars+=",DISTILL_BATCH_SIZE=${DISTILL_BATCH_SIZE}"
      export_vars+=",MEAN_FLOW_DIAG_FRACTION=${MEAN_FLOW_DIAG_FRACTION}"
      export_vars+=",MEAN_FLOW_ADAPTIVE_P=${MEAN_FLOW_ADAPTIVE_P}"
      export_vars+=",MEAN_FLOW_TIME_MU=${MEAN_FLOW_TIME_MU}"
      export_vars+=",MEAN_FLOW_TIME_SIGMA=${MEAN_FLOW_TIME_SIGMA}"
      export_vars+=",MEAN_FLOW_MAX_GRAD_NORM=${MEAN_FLOW_MAX_GRAD_NORM}"
      export_vars+=",MEAN_FLOW_USE_ADAMW=${MEAN_FLOW_USE_ADAMW}"
      export_vars+=",MEAN_FLOW_ADAM_BETA2=${MEAN_FLOW_ADAM_BETA2}"
      export_vars+=",CLASS_CONDITIONAL=${CLASS_CONDITIONAL}"
      export_vars+=",LABEL_DROPOUT=${LABEL_DROPOUT},GUIDANCE_SCALE=${GUIDANCE_SCALE}"
      export_vars+=",UNET_SIZE=${UNET_SIZE},MAX_GRAD_NORM=${MAX_GRAD_NORM}"
      export_vars+=",N_EXPERIENCES=${N_EXPERIENCES}"
      export_vars+=",PER_CLASS_FID_SAMPLES=${PER_CLASS_FID_SAMPLES}"
      export_vars+=",NO_KLD=${NO_KLD}"
      if [[ -n "${FID_TEST_SAMPLES}" ]]; then
        export_vars+=",FID_TEST_SAMPLES=${FID_TEST_SAMPLES}"
      fi
      if [[ -n "${FID_VAL_SAMPLES}" ]]; then
        export_vars+=",FID_VAL_SAMPLES=${FID_VAL_SAMPLES}"
      fi
      if [[ -n "${MAX_TASKS}" ]]; then
        export_vars+=",MAX_TASKS=${MAX_TASKS}"
      fi
      if [[ -n "${LR}" ]]; then
        export_vars+=",LR=${LR}"
      fi
      if [[ -n "${LATENT_DIR:-}" ]]; then
        export_vars+=",LATENT_DIR=${LATENT_DIR}"
      fi
      sbatch_args=()
      if [[ -n "${QOS}" ]]; then
        sbatch_args+=(--qos="${QOS}")
      fi
      if [[ -n "${TIME}" ]]; then
        sbatch_args+=(--time="${TIME}")
      fi
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  DRY_RUN sbatch --job-name=${job} ${sbatch_args[*]-} --export=${export_vars} ${SBATCH}"
      else
        sbatch --job-name="${job}" "${sbatch_args[@]}" --export="${export_vars}" "${SBATCH}"
      fi
      n=$((n + 1))
    done
  done
done
echo "Submitted ${n} jobs (DRY_RUN=${DRY_RUN})."
