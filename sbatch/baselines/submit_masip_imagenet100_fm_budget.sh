#!/usr/bin/env bash
# Masip Split-ImageNet-100 diffusion baselines re-run at the *flow-matching* budget.
#
# The original ImageNet Masip grid used the upstream medium UNet
# (128,256,256,256), 200 epochs, batch 128, lr 2e-4 — i.e. a smaller model and a
# shorter schedule than the locked FM recipe. This submitter swaps in
# ``baselines/configs/imagenet100_fm_budget/`` so the diffusion rows match the FM
# rows on capacity and optimizer steps:
#
#   UNet large (128,256,512,512), attn head dim 64, Adam lr 1e-4, grad clip 1.0
#   100,050 target Adam steps/task (the locked FM task-0 budget).
#   Masip computes the required epoch count per experience from its actual
#   dataloader size. This also keeps cumulative at ~100k updates as its union grows.
#
# Why the replay rows use batch 128: Masip appends an equally sized replay batch
# to every real batch, while FM keeps the total batch at 256 and splits it
# real/replay by λ. Batch 128 real + 128 replay = 256 samples/step, matching FM's
# per-step sample budget at the same 100k steps.
#
# Results land under a separate OUTPUT_DIR so the old medium-budget numbers stay
# on disk (upstream skips any run dir that already has completed.txt).
#
# From repo root:
#   ./sbatch/baselines/submit_masip_imagenet100_fm_budget.sh
#   DRY_RUN=1 ./sbatch/baselines/submit_masip_imagenet100_fm_budget.sh
#   SEEDS="42" STRATEGIES="naive" ./sbatch/baselines/submit_masip_imagenet100_fm_budget.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(${SEEDS:-42 69 1714})
STRATEGIES=(${STRATEGIES:-naive cumulative no_distillation full_generation_distillation})

CONFIG_DIR="${CONFIG_DIR:-${REPO_ROOT}/baselines/configs/imagenet100_fm_budget}"
GENERATOR_CONFIG="${GENERATOR_CONFIG:-${CONFIG_DIR}/ddim_large_4ch_latent.json}"

DATA_CACHE="${DATA_CACHE:-${REPO_ROOT}/data_dir/cache}"
# Separate root: run dir names are identical to the medium-budget grid.
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results_dir/baselines/masip_fm_budget}"
LATENT_DIR="${LATENT_DIR:-${REPO_ROOT}/data_dir/imagenet100/latents_sdvae_ft_mse_256}"
VAE_ID="${VAE_ID:-stabilityai/sd-vae-ft-mse}"

# FM-matched protocol knobs (unchanged from the medium-budget grid).
GENERATION_STEPS="${GENERATION_STEPS:-50}"
LAMBD="${LAMBD:-0.75}"
CLASS_CONDITIONAL="${CLASS_CONDITIONAL:-1}"
N_EXPERIENCES="${N_EXPERIENCES:-3}"
LABEL_DROPOUT="${LABEL_DROPOUT:-0.1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
SOLVER_TYPE="${SOLVER_TYPE:-None}"
DISABLE_FUSED_SDPA="${DISABLE_FUSED_SDPA:-1}"
# Per-experience Avalanche checkpoints: lets a walltime kill resume at the next task.
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"

QOS="${QOS:-YOUR_QOS}"
TIME="${TIME:-96:00:00}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"
JOB_PREFIX="${JOB_PREFIX:-masipfb-in100}"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_masip_cl.sbatch"

echo "=== Masip split_imagenet100 @ FM budget ==="
echo "STRATEGIES=${STRATEGIES[*]}"
echo "SEEDS=${SEEDS[*]}"
echo "GENERATOR_CONFIG=${GENERATOR_CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "QOS=${QOS} TIME=${TIME} MEM=${MEM} GRES=${GRES}"

strategy_config_for() {
  case "$1" in
    naive) echo "${CONFIG_DIR}/diffusion_naive.json" ;;
    cumulative) echo "${CONFIG_DIR}/diffusion_cumulative.json" ;;
    no_distillation) echo "${CONFIG_DIR}/diffusion_no_distill.json" ;;
    full_generation_distillation) echo "${CONFIG_DIR}/diffusion_full_gen_distill.json" ;;
    *) echo "" ;;
  esac
}

n_sub=0
for strategy in "${STRATEGIES[@]}"; do
  strategy_config="$(strategy_config_for "${strategy}")"
  if [[ -z "${strategy_config}" || ! -f "${strategy_config}" ]]; then
    echo "ERROR: no FM-budget config for STRATEGY=${strategy}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    job_name="${JOB_PREFIX}-${strategy:0:8}-s${seed}"
    export_vars="ALL,DATASET=split_imagenet100,STRATEGY=${strategy_config},SEED=${seed}"
    export_vars+=",GENERATOR_CONFIG=${GENERATOR_CONFIG},LAMBD=${LAMBD}"
    export_vars+=",GENERATION_STEPS=${GENERATION_STEPS},SOLVER_TYPE=${SOLVER_TYPE}"
    export_vars+=",KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS},DISABLE_FUSED_SDPA=${DISABLE_FUSED_SDPA}"
    export_vars+=",OUTPUT_DIR=${OUTPUT_DIR},DATA_CACHE=${DATA_CACHE}"
    export_vars+=",REPO_ROOT=${REPO_ROOT}"
    export_vars+=",LATENT_DIR=${LATENT_DIR},VAE_ID=${VAE_ID}"
    export_vars+=",CLASS_CONDITIONAL=${CLASS_CONDITIONAL},N_EXPERIENCES=${N_EXPERIENCES}"
    export_vars+=",LABEL_DROPOUT=${LABEL_DROPOUT},GUIDANCE_SCALE=${GUIDANCE_SCALE}"
    export_vars+=",PYTHONUNBUFFERED=1"

    echo "Submit: ${job_name}  (${strategy_config##*/})"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "  DRY_RUN sbatch --job-name=${job_name} --qos=${QOS} --time=${TIME} --mem=${MEM} --gres=${GRES} --export=${export_vars} ${SBATCH_SCRIPT}"
    else
      sbatch --job-name="${job_name}" --qos="${QOS}" --time="${TIME}" \
        --mem="${MEM}" --gres="${GRES}" \
        --export="${export_vars}" "${SBATCH_SCRIPT}"
    fi
    n_sub=$((n_sub + 1))
  done
done

echo "Submitted ${n_sub} jobs (DRY_RUN=${DRY_RUN})."
echo "Logs: ${SBATCH_ROOT}/logs/masip_cl_*.log"
echo "Outputs: ${OUTPUT_DIR}/split_imagenet100/"
