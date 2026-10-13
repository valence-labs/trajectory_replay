#!/usr/bin/env bash
# FMNIST trajectory-replay teacher-resolution sweep.
#
# Reuses the canonical NFE=10 runs:
#   results_dir/image_cl/fmnist/trajectory_eq_s{42,69,1714}
#
# Submits NFE={5,20,40,80,100} × 3 seeds = 15 new jobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT="${REPO_ROOT}/sbatch/image_cl/submit_fmnist_min.sh"

SEEDS="${SEEDS:-42 69 1714}"
NFE_GRID=(${NFE_GRID:-5 20 40 80 100})
TIME="${TIME:-24:00:00}"
QOS="${QOS:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "=== FMNIST trajectory replay NFE sweep ==="
echo "NFE_GRID=${NFE_GRID[*]} SEEDS=${SEEDS}"
echo "LAMBDA_MODE=equal TIME=${TIME} QOS=${QOS:-default}"

submitted=0
for nfe in "${NFE_GRID[@]}"; do
  step_size="$(
    python3 - "${nfe}" <<'PY'
import sys
print(f"{1.0 / int(sys.argv[1]):.12g}")
PY
  )"
  echo "--- replay NFE=${nfe} step_size=${step_size} ---"
  STRATEGIES="trajectory" \
  SEEDS="${SEEDS}" \
  REPLAY_BACKENDS="ode" \
  LAMBDA_MODE="equal" \
  REPLAY_ODE_STEP_SIZE="${step_size}" \
  RUN_SUFFIX="replaynfe${nfe}" \
  TIME="${TIME}" \
  QOS="${QOS}" \
  DRY_RUN="${DRY_RUN}" \
    "${SUBMIT}"
  submitted=$((submitted + 3))
done

echo "Submitted ${submitted} jobs (DRY_RUN=${DRY_RUN})."
