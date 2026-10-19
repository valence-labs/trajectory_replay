#!/bin/bash
#SBATCH --job-name=simmix-reeval
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/sim_mixture_reeval_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Re-evaluate existing checkpoints (mean_loglik / energy_mixture).
#
#   sbatch --export=ALL sbatch/sim_mixture/reeval_metrics.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${EPOCHS:=400}"
: "${DEVICE:=cuda}"
: "${N_GEN:=4000}"
: "${METHODS:=trajectory generative oracle}"
: "${ODE_STEP_SIZE:=0.05}"

echo "Job ${SLURM_JOB_ID:-local} reeval at $(date -Is)"
uv sync
uv run python experiments/sim_mixture/reeval_metrics.py \
  --dim "${DIM}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --n-gen "${N_GEN}" \
  --ode-step-size "${ODE_STEP_SIZE}" \
  --methods ${METHODS}

echo "Finished at $(date -Is)"
