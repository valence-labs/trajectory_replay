#!/usr/bin/env bash
# Split-FMNIST MeanFlow replay table grid (zhuyu CIFAR schedule).
#
# 24 jobs: {generative, trajectory} × {λ-grid, λ-equal} × {N=1, N=4} × 3 seeds.
# Main FID/KLD stay ODE on the CFM (same eval as the ODE rows). N is the
# MeanFlow replay jump count during training, not the reported sampler.
#
#   DRY_RUN=1 ./sbatch/image_cl/submit_fmnist_meanflow.sh
#   ./sbatch/image_cl/submit_fmnist_meanflow.sh
#   MEAN_FLOW_STEPS_GRID="1" ./sbatch/image_cl/submit_fmnist_meanflow.sh
#
# Results:
#   results_dir/image_cl/fmnist/{generative,trajectory}_mf_n{1,4}[_eq]_s{seed}/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET="${DATASET:-split_fmnist}"
export RESULTS_SLUG="${RESULTS_SLUG:-fmnist}"
export STRATEGIES="${STRATEGIES:-generative trajectory}"
export SEEDS="${SEEDS:-42 69 1714}"
export REPLAY_BACKENDS="mean_flow"
export LAMBDA_GRID="${LAMBDA_GRID:-0.1 0.25 0.5 0.75 0.9}"
# zhuyu CIFAR MeanFlow (ratio_r_not_equal_t=0.75 → diag=0.25). Its μ=-2
# data→noise schedule is mirrored to μ=+2 for our noise→data convention.
# adaptive_p=0.75 applies to task 0 only; joint CL stays unweighted so λ mixes
# comparable squared errors.
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
TIME_GRID="${TIME_GRID:-48:00:00}"
TIME_EQUAL="${TIME_EQUAL:-24:00:00}"
# Optional single walltime for every combo (overrides TIME_GRID / TIME_EQUAL).
TIME_ALL="${TIME_ALL:-}"

echo "=== FMNIST MeanFlow table grid ==="
echo "STRATEGIES=${STRATEGIES}"
echo "SEEDS=${SEEDS}"
echo "N=${MEAN_FLOW_STEPS_GRID[*]} LAMBDA_MODES=${LAMBDA_MODES[*]}"
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
    echo "--- N=${nfe} LAMBDA_MODE=${mode} TIME=${time_this} ---"
    MEAN_FLOW_STEPS="${nfe}" LAMBDA_MODE="${mode}" TIME="${time_this}" \
      "${REPO_ROOT}/sbatch/image_cl/submit_fmnist_min.sh"
    n_batches=$((n_batches + 1))
  done
done
echo "Launched ${n_batches} submit batches (DRY_RUN=${DRY_RUN})."
