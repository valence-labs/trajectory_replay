#!/usr/bin/env bash
# Minimal Masip FMNIST paper grid:
#   naive / cumulative (joint upper bound) / no_distillation / full_generation_distillation
# One SLURM job per (strategy, seed). Paper seeds: 42, 69, 1714.
#
# From repo root:
#   ./sbatch/baselines/submit_masip_fmnist_min.sh
#   DRY_RUN=1 ./sbatch/baselines/submit_masip_fmnist_min.sh
#   SEEDS="42" STRATEGIES="naive cumulative" ./sbatch/baselines/submit_masip_fmnist_min.sh
#   KEEP_CHECKPOINTS=1 LAMBD=1.0 ./sbatch/baselines/submit_masip_fmnist_min.sh
#
# LAMBD default 0.75 = paper Table B.4 generative distillation (2 teacher steps).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(${SEEDS:-42 69 1714})
STRATEGIES=(${STRATEGIES:-naive cumulative no_distillation full_generation_distillation})
DATASET="${DATASET:-split_fmnist}"
LAMBD="${LAMBD:-0.75}"
QOS="${QOS:-}"
TIME="${TIME:-}"
case "${DATASET}" in
  split_cifar10)
    GENERATION_STEPS="${GENERATION_STEPS:-20}"
    JOB_PREFIX="${JOB_PREFIX:-masip-c10}"
    SOLVER_TYPE="${SOLVER_TYPE:-cnn}"
    ;;
  split_imagenet100)
    GENERATION_STEPS="${GENERATION_STEPS:-20}"
    JOB_PREFIX="${JOB_PREFIX:-masip-in100}"
    SOLVER_TYPE="${SOLVER_TYPE:-None}"
    TIME="${TIME:-48:00:00}"
    QOS="${QOS:-YOUR_QOS}"
    ;;
  *)
    GENERATION_STEPS="${GENERATION_STEPS:-10}"
    JOB_PREFIX="${JOB_PREFIX:-masip-fm}"
    SOLVER_TYPE="${SOLVER_TYPE:-cnn}"
    ;;
esac
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
# Required on H100: torch 2.0.1 fused attention kernels miscompile on sm_90.
DISABLE_FUSED_SDPA="${DISABLE_FUSED_SDPA:-1}"

DATA_CACHE="${DATA_CACHE:-${REPO_ROOT}/data_dir/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results_dir/baselines/masip}"
LATENT_DIR="${LATENT_DIR:-${REPO_ROOT}/data_dir/imagenet100/latents_sdvae_ft_mse_256}"
VAE_ID="${VAE_ID:-stabilityai/sd-vae-ft-mse}"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_masip_cl.sbatch"

echo "=== Masip ${DATASET} minimal grid ==="
echo "STRATEGIES=${STRATEGIES[*]}"
echo "SEEDS=${SEEDS[*]} LAMBD=${LAMBD} GENERATION_STEPS=${GENERATION_STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATA_CACHE=${DATA_CACHE}"
echo "KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS} SOLVER_TYPE=${SOLVER_TYPE} DISABLE_FUSED_SDPA=${DISABLE_FUSED_SDPA}"
echo "QOS=${QOS:-default} TIME=${TIME:-sbatch-default}"

n_sub=0
for strategy in "${STRATEGIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    job_name="${JOB_PREFIX}-${strategy:0:8}-s${seed}"
    export_vars="ALL,DATASET=${DATASET},STRATEGY=${strategy},SEED=${seed},LAMBD=${LAMBD}"
    export_vars+=",GENERATION_STEPS=${GENERATION_STEPS},SOLVER_TYPE=${SOLVER_TYPE}"
    export_vars+=",KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS},DISABLE_FUSED_SDPA=${DISABLE_FUSED_SDPA}"
    export_vars+=",OUTPUT_DIR=${OUTPUT_DIR},DATA_CACHE=${DATA_CACHE}"
    export_vars+=",REPO_ROOT=${REPO_ROOT}"
    if [[ "${DATASET}" == "split_imagenet100" ]]; then
      export_vars+=",LATENT_DIR=${LATENT_DIR},VAE_ID=${VAE_ID}"
    fi
    # Class-conditional ImageNet protocol (optional elsewhere).
    if [[ -n "${CLASS_CONDITIONAL:-}" ]]; then
      export_vars+=",CLASS_CONDITIONAL=${CLASS_CONDITIONAL}"
    fi
    if [[ -n "${N_EXPERIENCES:-}" ]]; then
      export_vars+=",N_EXPERIENCES=${N_EXPERIENCES}"
    fi
    if [[ -n "${LABEL_DROPOUT:-}" ]]; then
      export_vars+=",LABEL_DROPOUT=${LABEL_DROPOUT}"
    fi
    if [[ -n "${GUIDANCE_SCALE:-}" ]]; then
      export_vars+=",GUIDANCE_SCALE=${GUIDANCE_SCALE}"
    fi

    sbatch_args=()
    if [[ -n "${QOS}" ]]; then
      sbatch_args+=(--qos="${QOS}")
    fi
    if [[ -n "${TIME}" ]]; then
      sbatch_args+=(--time="${TIME}")
    fi
    if [[ "${DATASET}" == "split_imagenet100" ]]; then
      sbatch_args+=(--mem=64G)
    fi

    echo "Submit: ${job_name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "  DRY_RUN sbatch --job-name=${job_name} ${sbatch_args[*]-} --export=${export_vars} ${SBATCH_SCRIPT}"
    else
      sbatch --job-name="${job_name}" "${sbatch_args[@]}" --export="${export_vars}" "${SBATCH_SCRIPT}"
    fi
    n_sub=$((n_sub + 1))
  done
done

echo "Submitted ${n_sub} jobs (DRY_RUN=${DRY_RUN})."
echo "Logs: ${SBATCH_ROOT}/logs/masip_cl_*.log"
echo "Outputs: ${OUTPUT_DIR}/${DATASET}/"
