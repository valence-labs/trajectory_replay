#!/usr/bin/env bash
# Split-ImageNet-100 MeanFlow replay table grid (zhuyu ImageNet schedule).
#
# Same 24-job layout as CIFAR-10 MeanFlow, on the locked class-cond recipe:
#   {generative, trajectory} × {λ-grid, λ-equal} × {N=1, N=4} × 3 seeds.
# λ-grid is {0.25, 0.5, 0.75} (same as the ODE g3 rows). FID stays ODE on the
# CFM (eval step 0.02); N is the MeanFlow replay jump count during training.
#
#   DRY_RUN=1 ./sbatch/image_cl/submit_imagenet100_meanflow.sh
#   ./sbatch/image_cl/submit_imagenet100_meanflow.sh
#   MEAN_FLOW_STEPS_GRID="4" ./sbatch/image_cl/submit_imagenet100_meanflow.sh
#
# Results:
#   results_dir/image_cl/imagenet100/{generative,trajectory}_mf_n{1,4}_g3_s{seed}/
#   results_dir/image_cl/imagenet100/{generative,trajectory}_mf_n{1,4}_eq_s{seed}/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET=split_imagenet100
export RESULTS_SLUG="${RESULTS_SLUG:-imagenet100}"
export STRATEGIES="${STRATEGIES:-generative trajectory}"
export SEEDS="${SEEDS:-42 69 1714}"
export REPLAY_BACKENDS="mean_flow"
export LAMBDA_GRID="${LAMBDA_GRID:-0.25 0.5 0.75}"
# zhuyu ImageNet MeanFlow (ratio_r_not_equal_t=0.25 → diag=0.75). μ=-0.4
# data→noise is mirrored to μ=+0.4 for our noise→data convention.
export MEAN_FLOW_DIAG_FRACTION="${MEAN_FLOW_DIAG_FRACTION:-0.75}"
export MEAN_FLOW_ADAPTIVE_P="${MEAN_FLOW_ADAPTIVE_P:-1.0}"
export MEAN_FLOW_TIME_MU="${MEAN_FLOW_TIME_MU:-0.4}"
export MEAN_FLOW_TIME_SIGMA="${MEAN_FLOW_TIME_SIGMA:-1.0}"
export MEAN_FLOW_MAX_GRAD_NORM="${MEAN_FLOW_MAX_GRAD_NORM:-1.0}"
export MEAN_FLOW_USE_ADAMW="${MEAN_FLOW_USE_ADAMW:-1}"
export MEAN_FLOW_ADAM_BETA2="${MEAN_FLOW_ADAM_BETA2:-0.95}"
export QOS="${QOS:-YOUR_QOS}"
export DRY_RUN="${DRY_RUN:-0}"

MEAN_FLOW_STEPS_GRID=(${MEAN_FLOW_STEPS_GRID:-1 4})
LAMBDA_MODES=(${LAMBDA_MODES:-grid equal})
# ODE ImageNet traj/gene λ-grid is ~17.5 h / λ @ 100k steps (6 later-task λ +
# task 0). Joint MeanFlow is a bit slower than ODE teacher eval; 7d / 4d leave
# headroom vs a ~22 h / λ guess. Override with TIME_ALL if needed.
TIME_GRID="${TIME_GRID:-168:00:00}"
TIME_EQUAL="${TIME_EQUAL:-96:00:00}"
TIME_ALL="${TIME_ALL:-}"

echo "=== ImageNet-100 MeanFlow table grid ==="
echo "STRATEGIES=${STRATEGIES}"
echo "SEEDS=${SEEDS}"
echo "N=${MEAN_FLOW_STEPS_GRID[*]} LAMBDA_MODES=${LAMBDA_MODES[*]}"
echo "LAMBDA_GRID=${LAMBDA_GRID}"
echo "diag=${MEAN_FLOW_DIAG_FRACTION} adaptive_p=${MEAN_FLOW_ADAPTIVE_P} μ=${MEAN_FLOW_TIME_MU} σ=${MEAN_FLOW_TIME_SIGMA}"
echo "TIME_GRID=${TIME_GRID} TIME_EQUAL=${TIME_EQUAL} TIME_ALL=${TIME_ALL:-none} QOS=${QOS}"

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
    # Grid names get _g3 like the ODE ImageNet table rows; equal gets _eq
    # from LAMBDA_MODE (do not also pass RUN_SUFFIX=eq → eq_eq).
    if [[ "${mode}" == "grid" ]]; then
      suffix="g3"
    else
      suffix=""
    fi
    echo "--- N=${nfe} LAMBDA_MODE=${mode} RUN_SUFFIX=${suffix:-none} TIME=${time_this} ---"
    MEAN_FLOW_STEPS="${nfe}" LAMBDA_MODE="${mode}" TIME="${time_this}" \
      RUN_SUFFIX="${suffix}" \
      "${REPO_ROOT}/sbatch/image_cl/submit_imagenet100_min.sh"
    n_batches=$((n_batches + 1))
  done
done
echo "Launched ${n_batches} submit batches (DRY_RUN=${DRY_RUN})."
