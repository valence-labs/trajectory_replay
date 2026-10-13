#!/usr/bin/env bash
# Submit the MeanFlow sample-scatter renderer as a batch job (never on login node).
#
#   N_STEPS="1 2 4 8" ./sbatch/sim_mixture/submit_mean_flow_plots.sh
#   DRY_RUN=1 ./sbatch/sim_mixture/submit_mean_flow_plots.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DIM="${DIM:-2}"
export GLOB="${GLOB:-ft_*_mf_e*_seed*}"
export N_STEPS="${N_STEPS:-4}"
export N_GEN="${N_GEN:-4000}"
export SEED="${SEED:-1999}"
export DEVICE="${DEVICE:-cuda}"

cmd=(
  sbatch
  --export=ALL
  --parsable
  --job-name="simmix-mfplot"
  --output="${SBATCH_ROOT}/logs/sim_mixture_mfplot_%j.log"
  "${SCRIPT_DIR}/plot_mean_flow_samples.sbatch"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "${cmd[*]}  # DIM=${DIM} GLOB=${GLOB} N_STEPS=${N_STEPS}"
else
  jid="$("${cmd[@]}")"
  echo "Submitted MeanFlow plot job=${jid} (N_STEPS=${N_STEPS})" >&2
  echo "Log: ${SBATCH_ROOT}/logs/sim_mixture_mfplot_${jid}.log"
fi
