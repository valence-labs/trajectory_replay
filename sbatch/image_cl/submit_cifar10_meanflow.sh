#!/usr/bin/env bash
# Split-CIFAR-10 MeanFlow replay table grid (zhuyu CIFAR schedule).
#
# Matches the ODE CIFAR table: λ-grid {0.25, 0.5, 0.75} (same as generative_g3 /
# trajectory_g3), λ-equal, N=1 and N=4 replay jumps.
# 24 jobs: {generative, trajectory} × {λ-grid, λ-equal} × {N=1, N=4} × 3 seeds.
# Main FID/KLD stay ODE on the CFM. N is the MeanFlow replay jump count.
#
#   DRY_RUN=1 ./sbatch/image_cl/submit_cifar10_meanflow.sh
#   ./sbatch/image_cl/submit_cifar10_meanflow.sh
#   MEAN_FLOW_STEPS_GRID="4" ./sbatch/image_cl/submit_cifar10_meanflow.sh
#
# Results:
#   results_dir/image_cl/cifar10/{generative,trajectory}_mf_n{1,4}_g3_s{seed}/
#   results_dir/image_cl/cifar10/{generative,trajectory}_mf_n{1,4}_eq_s{seed}/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET=split_cifar10
export RESULTS_SLUG="${RESULTS_SLUG:-cifar10}"
export STRATEGIES="${STRATEGIES:-generative trajectory}"
export SEEDS="${SEEDS:-42 69 1714}"
export REPLAY_BACKENDS="mean_flow"
# Same 3-λ sweep as the ODE CIFAR g3 rows in tab:cifar10_cl.
export LAMBDA_GRID="${LAMBDA_GRID:-0.25 0.5 0.75}"
# zhuyu CIFAR MeanFlow (ratio_r_not_equal_t=0.75 → diag=0.25). μ=-2
# data→noise is mirrored to μ=+2 for our noise→data convention.
export MEAN_FLOW_DIAG_FRACTION="${MEAN_FLOW_DIAG_FRACTION:-0.25}"
export MEAN_FLOW_ADAPTIVE_P="${MEAN_FLOW_ADAPTIVE_P:-0.75}"
export MEAN_FLOW_TIME_MU="${MEAN_FLOW_TIME_MU:-2}"
export MEAN_FLOW_TIME_SIGMA="${MEAN_FLOW_TIME_SIGMA:-2}"
export MEAN_FLOW_MAX_GRAD_NORM="${MEAN_FLOW_MAX_GRAD_NORM:-1.0}"
export MEAN_FLOW_USE_ADAMW="${MEAN_FLOW_USE_ADAMW:-1}"
export MEAN_FLOW_ADAM_BETA2="${MEAN_FLOW_ADAM_BETA2:-0.95}"
export QOS="${QOS:-}"
export DRY_RUN="${DRY_RUN:-0}"

MEAN_FLOW_STEPS_GRID=(${MEAN_FLOW_STEPS_GRID:-1 4})
LAMBDA_MODES=(${LAMBDA_MODES:-grid equal})
# CIFAR ODE g3 is ~1.5–2.5 h / λ / task; MeanFlow JVP is slower. Grid = 3λ × 4
# later tasks; equal = 1λ × 4. Limits leave headroom vs ODE walls.
TIME_GRID="${TIME_GRID:-72:00:00}"
TIME_EQUAL="${TIME_EQUAL:-48:00:00}"
TIME_ALL="${TIME_ALL:-}"

echo "=== CIFAR-10 MeanFlow table grid ==="
echo "STRATEGIES=${STRATEGIES}"
echo "SEEDS=${SEEDS}"
echo "N=${MEAN_FLOW_STEPS_GRID[*]} LAMBDA_MODES=${LAMBDA_MODES[*]}"
echo "LAMBDA_GRID=${LAMBDA_GRID}"
echo "diag=${MEAN_FLOW_DIAG_FRACTION} adaptive_p=${MEAN_FLOW_ADAPTIVE_P} μ=${MEAN_FLOW_TIME_MU} σ=${MEAN_FLOW_TIME_SIGMA}"
echo "TIME_GRID=${TIME_GRID} TIME_EQUAL=${TIME_EQUAL} TIME_ALL=${TIME_ALL:-none}"

n_batches=0
for nfe in "${MEAN_FLOW_STEPS_GRID[@]}"; do
  for mode in "${LAMBDA_MODES[@]}"; do
    if [[ -n "${TIME_ALL}" ]]; then
      time_this="${TIME_ALL}"
    elif [[ "${mode}" == "equal" ]]; then
      time_this="${TIME_EQUAL}"
    else
      time_this="${TIME_GRID}"
    fi
    # Grid names get _g3 like the ODE CIFAR table rows; equal gets _eq.
    if [[ "${mode}" == "grid" ]]; then
      suffix="g3"
    else
      suffix=""
    fi
    echo "--- N=${nfe} LAMBDA_MODE=${mode} RUN_SUFFIX=${suffix:-none} TIME=${time_this} ---"
    MEAN_FLOW_STEPS="${nfe}" LAMBDA_MODE="${mode}" TIME="${time_this}" \
      RUN_SUFFIX="${suffix}" \
      "${REPO_ROOT}/sbatch/image_cl/submit_cifar10_min.sh"
    n_batches=$((n_batches + 1))
  done
done
echo "Launched ${n_batches} submit batches (DRY_RUN=${DRY_RUN})."
