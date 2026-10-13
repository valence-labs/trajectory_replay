#!/usr/bin/env bash
# Submit Split-CIFAR10 image_cl grid: strategies × seeds [× replay backends] [× λ modes].
#
# Default job set (9): naive / generative / trajectory × ODE × λ-grid × 3 seeds.
#
#   ./sbatch/image_cl/submit_cifar10_min.sh
#   STRATEGIES="trajectory" SEEDS="42" ./sbatch/image_cl/submit_cifar10_min.sh
#   # Expand to 15 jobs (ODE + MeanFlow replay on gen/traj):
#   REPLAY_BACKENDS="ode mean_flow" ./sbatch/image_cl/submit_cifar10_min.sh
#   # Fixed equal-weight λ=(n-1)/n (no sweep):
#   STRATEGIES="generative trajectory" LAMBDA_MODE=equal \
#     ./sbatch/image_cl/submit_cifar10_min.sh
#   # Full MeanFlow table (N=1 and N=4, g3 λ-grid + equal):
#   ./sbatch/image_cl/submit_cifar10_meanflow.sh
#   DRY_RUN=1 ./sbatch/image_cl/submit_cifar10_min.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"


export DATASET=split_cifar10
export RESULTS_SLUG="${RESULTS_SLUG:-cifar10}"
# Masip CIFAR defaults (overridable).
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.05}"
# Match FMNIST default grid.
export STRATEGIES="${STRATEGIES:-naive generative trajectory}"
export SEEDS="${SEEDS:-42 69 1714}"
export REPLAY_BACKENDS="${REPLAY_BACKENDS:-ode}"
export LAMBDA_MODE="${LAMBDA_MODE:-grid}"

exec "${REPO_ROOT}/sbatch/image_cl/submit_fmnist_min.sh"
