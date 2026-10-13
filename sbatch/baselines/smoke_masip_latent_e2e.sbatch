#!/bin/bash
#SBATCH --job-name=masip-in100-e2e
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=sbatch/logs/masip_in100_e2e_%j.log

# End-to-end integration check of the latent Masip ImageNet-100 baseline:
# runs the real train_cl.py through the Avalanche loop with 1 epoch/experience
# and a small FID sample budget, exercising generative replay, the decoded
# FID/KLD hooks, and the sample-grid dump.
#
#   sbatch sbatch/baselines/smoke_masip_latent_e2e.sbatch
#   STRATEGY=naive sbatch sbatch/baselines/smoke_masip_latent_e2e.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

MASIP_ROOT="${MASIP_ROOT:-${REPO_ROOT}/baselines/diffusion_continual_learning}"
STRATEGY="${STRATEGY:-full_generation_distillation}"
LATENT_DIR="${LATENT_DIR:-${HOME}/sandbox/tx_analysis_data/imagenet100/latents_sdvae_ft_mse_256}"
VAE_ID="${VAE_ID:-stabilityai/sd-vae-ft-mse}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/sandbox/trajectory_replay/baselines/masip_smoke}"

SANDBOX_ROOT="${SANDBOX_ROOT:-${HOME}/sandbox}"
DATA_CACHE="${DATA_CACHE:-${SANDBOX_ROOT}/trajectory_replay/data_cache}"
export AVALANCHE_HOME="${AVALANCHE_HOME:-${DATA_CACHE}/avalanche}"
export TORCH_HOME="${TORCH_HOME:-${DATA_CACHE}/torch}"
export HF_HOME="${HF_HOME:-${HOME}/sandbox/tx_analysis_data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${AVALANCHE_HOME}" "${TORCH_HOME}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

export DISABLE_FUSED_SDPA="${DISABLE_FUSED_SDPA:-1}"
export MASIP_METRIC_N_SAMPLES="${MASIP_METRIC_N_SAMPLES:-256}"

unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

case "${STRATEGY}" in
  naive) SC="configs/strategy/imagenet100/_smoke_diffusion_naive.json" ;;
  *)     SC="configs/strategy/imagenet100/_smoke_diffusion_full_gen_distill.json" ;;
esac

echo "Job ${SLURM_JOB_ID:-local} Masip latent e2e at $(date -Is)"
echo "  STRATEGY=${STRATEGY} CONFIG=${SC}"
echo "  LATENT_DIR=${LATENT_DIR} METRIC_N_SAMPLES=${MASIP_METRIC_N_SAMPLES}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

rm -rf "${OUTPUT_DIR}"

cd "${MASIP_ROOT}"
uv sync
uv run python train_cl.py \
  --dataset split_imagenet100 \
  --image_size 32 \
  --generator_type diffusion \
  --generator_config_path configs/model/ddim_medium_4ch_latent.json \
  --generator_strategy_config_path "${SC}" \
  --solver_type None \
  --lambd 0.75 \
  --generation_steps 20 \
  --eta 0.0 \
  --seed 42 \
  --cuda 0 \
  --kld_clf_path torchvision:resnet50 \
  --latent_dir "${LATENT_DIR}" \
  --vae_id "${VAE_ID}" \
  --output_dir "${OUTPUT_DIR}"

echo "=== sample grids ==="
find "${OUTPUT_DIR}" -name '*.png' | sort
echo "=== stream metrics ==="
find "${OUTPUT_DIR}" -name 'eval_results.csv' -exec sh -c 'echo "--- $1"; head -3 "$1"' _ {} \;

echo "Finished at $(date -Is)"
