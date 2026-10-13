#!/usr/bin/env bash
# Minimal Masip Split-CIFAR10 paper grid:
#   naive / cumulative (joint upper bound) / no_distillation / full_generation_distillation
# One SLURM job per (strategy, seed). Paper seeds: 42, 69, 1714 -> 12 jobs.
#
# CIFAR differs from FMNIST: 200 epochs, batch 64, 3-channel ddim_medium_3ch
# backbone, 20 generation steps, configs under configs/strategy/cifar/, and the
# cnn_cifar10 KLD classifier. Replay strategies use teacher_steps=10 (paper main
# text / Table 1), via diffusion_no_distill_preliminary_10.json and
# diffusion_full_gen_distill_10.json — not the Table B.5 teacher=2 ablation.
#
# From repo root:
#   ./sbatch/baselines/submit_masip_cifar10_min.sh
#   DRY_RUN=1 ./sbatch/baselines/submit_masip_cifar10_min.sh
#   SEEDS="42" STRATEGIES="naive" ./sbatch/baselines/submit_masip_cifar10_min.sh
#   # Rerun only teacher-dependent strategies after the teacher_steps=10 fix:
#   STRATEGIES="no_distillation full_generation_distillation" \
#     ./sbatch/baselines/submit_masip_cifar10_min.sh
#
# LAMBD default 0.75 = paper Table B.4 / B.5 generative distillation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET=split_cifar10
export GENERATION_STEPS="${GENERATION_STEPS:-20}"
export JOB_PREFIX="${JOB_PREFIX:-masip-c10}"

exec "${SCRIPT_DIR}/submit_masip_fmnist_min.sh"
