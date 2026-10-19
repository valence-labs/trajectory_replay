#!/bin/bash
#SBATCH --job-name=masip-cl
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/masip_cl_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Masip et al. continual diffusion baseline (one seed per job).
#
# Submit from repo root, e.g.:
#   sbatch --export=ALL,STRATEGY=naive,DATASET=split_fmnist,SEED=42 \
#     sbatch/baselines/train_masip_cl.sbatch
#   sbatch --export=ALL,STRATEGY=cumulative,DATASET=split_fmnist,SEED=42 \
#     sbatch/baselines/train_masip_cl.sbatch
#   sbatch --export=ALL,STRATEGY=no_distillation,DATASET=split_fmnist,SEED=42 \
#     sbatch/baselines/train_masip_cl.sbatch
#   sbatch --export=ALL,STRATEGY=full_generation_distillation,DATASET=split_fmnist,SEED=42,LAMBD=0.75 \
#     sbatch/baselines/train_masip_cl.sbatch
#
# Or use: ./sbatch/baselines/submit_masip_fmnist_min.sh
#                ./sbatch/baselines/submit_masip_cifar10_min.sh
#
# Override env vars: STRATEGY, DATASET, SEED, LAMBD, GENERATION_STEPS,
#   OUTPUT_DIR, DATA_CACHE, KLD_CLF_PATH, SOLVER_TYPE, KEEP_CHECKPOINTS, TIME via sbatch -t.
# LAMBD default 0.75 = paper Table B.4 generative distillation (2 teacher steps).
# GENERATION_STEPS defaults per dataset: 10 (FMNIST/MNIST), 20 (CIFAR-10).

set -euo pipefail

# Slurm copies the batch script into its spool dir, so BASH_SOURCE is useless here.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
SBATCH_ROOT="${REPO_ROOT}/sbatch"
mkdir -p "${SBATCH_ROOT}/logs"

MASIP_ROOT="${MASIP_ROOT:-${REPO_ROOT}/baselines/diffusion_continual_learning}"
if [[ ! -f "${MASIP_ROOT}/train_cl.py" ]]; then
  echo "ERROR: Masip clone missing at ${MASIP_ROOT}" >&2
  echo "Clone/pin per baselines/README.md" >&2
  exit 1
fi

: "${DATASET:=split_fmnist}"
: "${STRATEGY:=naive}"
: "${SEED:=42}"
: "${LAMBD:=0.75}"
# GENERATION_STEPS default is dataset-dependent; set in the DATASET case below.
: "${ETA:=0.0}"
: "${IMAGE_SIZE:=32}"
: "${CUDA_IDX:=0}"
# SOLVER_TYPE default is dataset-dependent; applied after the DATASET case below.
: "${KEEP_CHECKPOINTS:=0}"
: "${WANDB:=0}"
# torch 2.0.1 fused attention kernels are broken on H100 (sm_90); see train_cl.py.
: "${DISABLE_FUSED_SDPA:=1}"

DATA_CACHE="${DATA_CACHE:-${REPO_ROOT}/data_dir/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results_dir/baselines/masip}"

# Dataset routing first: the generator strategy configs are dataset-specific
# (CIFAR uses 200 epochs / batch 64 under configs/strategy/cifar/).
case "${DATASET}" in
  split_fmnist)
    : "${GENERATOR_CONFIG:=configs/model/ddim_medium.json}"
    : "${SOLVER_CONFIG:=configs/model/cnn.json}"
    : "${SOLVER_STRATEGY_CONFIG:=configs/strategy/cnn_w_diffusion.json}"
    : "${KLD_CLF_PATH:=${MASIP_ROOT}/weights/cnn_fmnist/}"
    : "${GENERATION_STEPS:=10}"
    STRATEGY_CONFIG_DIR="configs/strategy"
    NO_DISTILL_CONFIG="diffusion_no_distill.json"
    ;;
  split_cifar10)
    : "${GENERATOR_CONFIG:=configs/model/ddim_medium_3ch.json}"
    : "${SOLVER_CONFIG:=configs/model/cnn_3ch.json}"
    : "${SOLVER_STRATEGY_CONFIG:=configs/strategy/cifar/cnn_w_diffusion.json}"
    : "${KLD_CLF_PATH:=${MASIP_ROOT}/weights/cnn_cifar10/}"
    # Eval DDIM steps: upstream launch scripts use 20 (paper main-text FID uses 10;
    # keep 20 so Joint/Finetune numbers stay comparable to prior runs).
    : "${GENERATION_STEPS:=20}"
    STRATEGY_CONFIG_DIR="configs/strategy/cifar"
    # Paper main text: 10 teacher DDIM steps on CIFAR-10 (2 is the FMNIST / Table B.5
    # ablation setting). Use the upstream *_10 configs, not preliminary_2 / default GD.
    NO_DISTILL_CONFIG="diffusion_no_distill_preliminary_10.json"
    FULL_DISTILL_CONFIG="diffusion_full_gen_distill_10.json"
    ;;
  split_imagenet100)
    # Latent-space Masip: trains on the same frozen SD-VAE 4x32x32 latents as
    # the FM runs and decodes to 256x256 for FID/KLD, so numbers are directly
    # comparable to the latent-FM rows.
    : "${GENERATOR_CONFIG:=configs/model/ddim_medium_4ch_latent.json}"
    : "${SOLVER_TYPE:=None}"
    : "${KLD_CLF_PATH:=torchvision:resnet50}"
    : "${GENERATION_STEPS:=20}"
    : "${IMAGE_SIZE:=32}"
    : "${LATENT_DIR:=${REPO_ROOT}/data_dir/imagenet100/latents_sdvae_ft_mse_256}"
    : "${VAE_ID:=stabilityai/sd-vae-ft-mse}"
    STRATEGY_CONFIG_DIR="configs/strategy/imagenet100"
    NO_DISTILL_CONFIG="diffusion_no_distill.json"
    FULL_DISTILL_CONFIG="diffusion_full_gen_distill.json"
    ;;
  split_mnist)
    : "${GENERATOR_CONFIG:=configs/model/ddim_medium.json}"
    : "${SOLVER_CONFIG:=configs/model/cnn.json}"
    : "${SOLVER_STRATEGY_CONFIG:=configs/strategy/cnn_w_diffusion.json}"
    : "${KLD_CLF_PATH:=${MASIP_ROOT}/weights/cnn_fmnist/}"
    : "${GENERATION_STEPS:=10}"
    STRATEGY_CONFIG_DIR="configs/strategy"
    NO_DISTILL_CONFIG="diffusion_no_distill.json"
    FULL_DISTILL_CONFIG="diffusion_full_gen_distill.json"
    ;;
  *)
    echo "ERROR: unknown DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

# Datasets with a classifier solver default to it; ImageNet-100 is generator-only.
: "${SOLVER_TYPE:=cnn}"
# Default full-GD config if a dataset case did not set one (FMNIST).
: "${FULL_DISTILL_CONFIG:=diffusion_full_gen_distill.json}"

case "${STRATEGY}" in
  naive)
    GENERATOR_STRATEGY_CONFIG="${STRATEGY_CONFIG_DIR}/diffusion_naive.json"
    ;;
  cumulative|joint)
    # Upper bound: train on all real data seen so far (joint / cumulative rehearsal).
    GENERATOR_STRATEGY_CONFIG="${STRATEGY_CONFIG_DIR}/diffusion_cumulative.json"
    ;;
  no_distillation|no_distill|generative)
    GENERATOR_STRATEGY_CONFIG="${STRATEGY_CONFIG_DIR}/${NO_DISTILL_CONFIG}"
    ;;
  full_generation_distillation|full_gd|gd)
    GENERATOR_STRATEGY_CONFIG="${STRATEGY_CONFIG_DIR}/${FULL_DISTILL_CONFIG}"
    ;;
  *)
    # Allow a relative path under the Masip clone, or an absolute path.
    if [[ -f "${STRATEGY}" ]]; then
      GENERATOR_STRATEGY_CONFIG="${STRATEGY}"
    elif [[ -f "${MASIP_ROOT}/${STRATEGY}" ]]; then
      GENERATOR_STRATEGY_CONFIG="${STRATEGY}"
    else
      echo "ERROR: unknown STRATEGY=${STRATEGY}" >&2
      echo "Use naive | cumulative | no_distillation | full_generation_distillation, or a config path." >&2
      exit 1
    fi
    ;;
esac

if [[ ! -f "${MASIP_ROOT}/${GENERATOR_STRATEGY_CONFIG}" && ! -f "${GENERATOR_STRATEGY_CONFIG}" ]]; then
  echo "ERROR: generator strategy config not found: ${GENERATOR_STRATEGY_CONFIG}" >&2
  echo "  (DATASET=${DATASET} STRATEGY=${STRATEGY} MASIP_ROOT=${MASIP_ROOT})" >&2
  exit 1
fi

if [[ "${KLD_CLF_PATH}" != torchvision:* && ! -d "${KLD_CLF_PATH}" ]]; then
  echo "ERROR: KLD classifier weights missing at ${KLD_CLF_PATH}" >&2
  echo "Train once with: cd ${MASIP_ROOT} && uv run python train_kld_classifier.py ..." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${DATA_CACHE}"

# Shared dataset / torch caches (avoid per-node re-downloads into $HOME).
export AVALANCHE_HOME="${AVALANCHE_HOME:-${DATA_CACHE}/avalanche}"
export TORCH_HOME="${TORCH_HOME:-${DATA_CACHE}/torch}"
# HuggingFace cache for the frozen SD-VAE used to decode ImageNet-100 latents.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data_dir/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${AVALANCHE_HOME}" "${TORCH_HOME}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

# Optional: retain Avalanche CheckpointPlugin dumps for debugging.
export KEEP_CHECKPOINTS
export DISABLE_FUSED_SDPA

# Avoid leaking Cursor sandbox UV cache paths from the submitter env (--export=ALL).
unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

cd "${MASIP_ROOT}"

extra=()
if [[ "${SOLVER_TYPE}" == "None" || "${SOLVER_TYPE}" == "none" ]]; then
  extra+=(--solver_type None)
else
  extra+=(--solver_type "${SOLVER_TYPE}")
  extra+=(--solver_config_path "${SOLVER_CONFIG}")
  extra+=(--solver_strategy_config_path "${SOLVER_STRATEGY_CONFIG}")
fi
if [[ "${WANDB}" == "1" ]]; then
  extra+=(--wandb)
fi

echo "Job ${SLURM_JOB_ID:-local} Masip CL at $(date -Is)"
echo "  MASIP_ROOT=${MASIP_ROOT}"
echo "  DATASET=${DATASET} STRATEGY=${STRATEGY} SEED=${SEED} LAMBD=${LAMBD}"
echo "  GENERATOR_STRATEGY_CONFIG=${GENERATOR_STRATEGY_CONFIG}"
echo "  GENERATION_STEPS=${GENERATION_STEPS} IMAGE_SIZE=${IMAGE_SIZE}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  AVALANCHE_HOME=${AVALANCHE_HOME}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "  KLD_CLF_PATH=${KLD_CLF_PATH}"
echo "  KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS} SOLVER_TYPE=${SOLVER_TYPE}"
echo "  DISABLE_FUSED_SDPA=${DISABLE_FUSED_SDPA}"
echo "  CLASS_CONDITIONAL=${CLASS_CONDITIONAL:-0} N_EXPERIENCES=${N_EXPERIENCES:-}"
echo "  LABEL_DROPOUT=${LABEL_DROPOUT:-0.1} GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.5}"
if [[ "${DATASET}" == "split_imagenet100" ]]; then
  echo "  LATENT_DIR=${LATENT_DIR} VAE_ID=${VAE_ID}"
fi
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: config resolution only, not training."
  exit 0
fi

uv sync
extra_img=()
if [[ "${DATASET}" == "split_imagenet100" ]]; then
  extra_img+=(--latent_dir "${LATENT_DIR}" --vae_id "${VAE_ID}")
fi
extra_cc=()
if [[ "${CLASS_CONDITIONAL:-0}" == "1" ]]; then
  extra_cc+=(--class_conditional)
  extra_cc+=(--label_dropout "${LABEL_DROPOUT:-0.1}")
  extra_cc+=(--guidance_scale "${GUIDANCE_SCALE:-1.5}")
fi
if [[ -n "${N_EXPERIENCES:-}" ]]; then
  extra_cc+=(--n_experiences "${N_EXPERIENCES}")
fi
if [[ -n "${NUM_CLASSES:-}" ]]; then
  extra_cc+=(--num_classes "${NUM_CLASSES}")
fi
uv run python train_cl.py \
  --dataset "${DATASET}" \
  --image_size "${IMAGE_SIZE}" \
  --generator_type diffusion \
  --generator_config_path "${GENERATOR_CONFIG}" \
  --generator_strategy_config_path "${GENERATOR_STRATEGY_CONFIG}" \
  --lambd "${LAMBD}" \
  --generation_steps "${GENERATION_STEPS}" \
  --eta "${ETA}" \
  --seed "${SEED}" \
  --cuda "${CUDA_IDX}" \
  --kld_clf_path "${KLD_CLF_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "${extra[@]}" \
  "${extra_img[@]}" \
  "${extra_cc[@]}"

echo "Finished at $(date -Is)"
echo "Results under: ${OUTPUT_DIR}/${DATASET}/"
