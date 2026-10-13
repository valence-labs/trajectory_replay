#!/bin/bash
#SBATCH --job-name=simmix-mfplot
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/sim_mixture_mfplot_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Render MeanFlow sample scatters at one or more N (sampling budgets).
#
#   N_STEPS="1 2 4 8" ./sbatch/sim_mixture/submit_mean_flow_plots.sh
#   sbatch --export=ALL,N_STEPS="4" sbatch/sim_mixture/plot_mean_flow_samples.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${GLOB:=ft_*_mf_e*_seed*}"
: "${N_STEPS:=4}"
: "${N_GEN:=4000}"
: "${SEED:=1999}"
: "${DEVICE:=cuda}"

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  DIM=${DIM}  GLOB=${GLOB}  N_STEPS=${N_STEPS}  DEVICE=${DEVICE}"

uv sync
uv run python experiments/sim_mixture/plot_mean_flow_samples.py \
  --dim "${DIM}" \
  --glob "${GLOB}" \
  --n-steps ${N_STEPS} \
  --n-gen "${N_GEN}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "Finished at $(date -Is)"
