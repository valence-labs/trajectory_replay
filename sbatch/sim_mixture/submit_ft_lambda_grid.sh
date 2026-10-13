#!/usr/bin/env bash
# Submit top-panel finetune grid from a shared pretrain checkpoint.
# SEEDS vary FT randomness only; PRETRAIN_CHECKPOINT stays fixed.
#
#   SEEDS="0 1 2" EPOCHS=200 ./sbatch/sim_mixture/submit_ft_lambda_grid.sh
#   DRY_RUN=1 ./sbatch/sim_mixture/submit_ft_lambda_grid.sh
#
# REPLAY_BACKEND=mean_flow co-trains the dual-time MeanFlow (two-branch identity)
# alongside the instantaneous field with the zhuyu CIFAR one-step schedule;
# those runs are tagged ``_mf_cifar``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
export SIM_ROOT
export DIM="${DIM:-2}"
export EPOCHS="${EPOCHS:-2000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-3}"
export DEVICE="${DEVICE:-cuda}"
export REPLAY_BACKEND="${REPLAY_BACKEND:-ode}"
export MEAN_FLOW_STEPS="${MEAN_FLOW_STEPS:-1}"
export DIAG_FRACTION="${DIAG_FRACTION:-0.75}"
export TIME_MU="${TIME_MU:-2.0}"
export TIME_SIGMA="${TIME_SIGMA:-2.0}"
export ADAPTIVE_P="${ADAPTIVE_P:-0}"
# zhuyu-cs/MeanFlow reference heuristics for the MeanFlow optimizer (EMA excluded).
export MF_MAX_GRAD_NORM="${MF_MAX_GRAD_NORM:-1.0}"
export MF_USE_ADAMW="${MF_USE_ADAMW:-1}"
export MF_ADAM_BETA2="${MF_ADAM_BETA2:-0.95}"
export MF_WEIGHT_DECAY="${MF_WEIGHT_DECAY:-0.0}"
[[ -n "${MF_LR:-}" ]] && export MF_LR

LAMBDAS=(${LAMBDAS:-0.1 0.3 0.5 0.7 0.9})
IFS=',' read -r -a STRATEGIES <<< "${STRATEGIES:-naive,generative,trajectory}"
SEEDS=(${SEEDS:-0})

# Shared frozen old-task model (not varied across FT seeds).
: "${PRETRAIN_CHECKPOINT:=${SIM_ROOT}/results_dir/sim_mixture/dim${DIM}/pretrain_task0_seed0/checkpoint.pt}"
export PRETRAIN_CHECKPOINT

DEP_OPT=()
if [[ -n "${DEPENDENCY:-}" ]]; then
  DEP_OPT=(--dependency="${DEPENDENCY}")
fi

submit_one() {
  local strategy="$1"
  local lam="${2:-}"
  local seed="$3"
  export STRATEGY="${strategy}"
  export TRAIN_SEED="${seed}"
  if [[ -n "${lam}" ]]; then
    export LAMBDA_MIX="${lam}"
  else
    export LAMBDA_MIX=0.5
  fi

  local tag="${strategy}"
  if [[ "${strategy}" != "naive" ]]; then
    tag="${strategy}_lam${lam}"
  fi
  if [[ "${REPLAY_BACKEND}" == "mean_flow" ]]; then
    local schedule_tag="mf_cifar"
    # CIFAR: σ=2 with |μ|=2; ImageNet: σ=1 with |μ|=0.4. Sign is flipped vs
    # zhuyu because this repo is noise→data (μ_repo = -μ_reference).
    case "${TIME_MU}:${TIME_SIGMA}" in
      0.4:1.0|+0.4:1.0|-0.4:1.0) schedule_tag="mf_imagenet" ;;
      2.0:2.0|+2.0:2.0|-2.0:2.0) schedule_tag="mf_cifar" ;;
      *)
        schedule_tag="mf_tmu${TIME_MU//./p}_tsig${TIME_SIGMA//./p}"
        schedule_tag="${schedule_tag//-/n}"
        ;;
    esac
    tag="${tag}_${schedule_tag}"
    if [[ "${DIAG_FRACTION}" != "0.75" ]]; then
      tag="${tag}_diag${DIAG_FRACTION//./p}"
    fi
    if [[ "${ADAPTIVE_P}" == "0" || "${ADAPTIVE_P}" == "0.0" ]]; then
      tag="${tag}_noadapt"
    fi
    if [[ -n "${MF_LR:-}" && "${MF_LR}" != "${LR}" ]]; then
      tag="${tag}_mflr${MF_LR}"
    fi
    if [[ "${MF_USE_ADAMW}" == "1" || "${MF_MAX_GRAD_NORM}" != "0.0" ]]; then
      tag="${tag}_adamw_b${MF_ADAM_BETA2//./p}_gc${MF_MAX_GRAD_NORM//./p}"
    fi
  fi
  export RUN_NAME="ft_${tag}_e${EPOCHS}_seed${seed}"

  local log_file="${SBATCH_ROOT}/logs/sim_mixture_${RUN_NAME//./p}_%j.log"
  local cmd=(
    sbatch
    --export=ALL
    --parsable
    --job-name="simmix-${tag//./p}-e${EPOCHS}-s${seed}"
    --output="${log_file}"
    "${DEP_OPT[@]}"
  )
  if [[ -n "${QOS:-}" ]]; then
    cmd+=(--qos="${QOS}")
  fi
  cmd+=("${SCRIPT_DIR}/train_finetune.sbatch")

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${cmd[*]}  # STRATEGY=${strategy} LAMBDA_MIX=${LAMBDA_MIX} SEED=${seed} PRETRAIN=${PRETRAIN_CHECKPOINT} RUN_NAME=${RUN_NAME}"
  else
    local jid
    jid="$("${cmd[@]}")"
    echo "Submitted: ${RUN_NAME} job=${jid}" >&2
  fi
}

if [[ "${DRY_RUN}" != "1" && ! -f "${PRETRAIN_CHECKPOINT}" ]]; then
  echo "ERROR: missing shared pretrain checkpoint: ${PRETRAIN_CHECKPOINT}" >&2
  exit 1
fi

echo "Using shared PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT}"

for seed in "${SEEDS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    strategy="$(echo "${strategy}" | tr -d '[:space:]')"
    if [[ "${strategy}" == "naive" ]]; then
      submit_one naive "" "${seed}"
    else
      for lam in "${LAMBDAS[@]}"; do
        submit_one "${strategy}" "${lam}" "${seed}"
      done
    fi
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Logs: ${SBATCH_ROOT}/logs/sim_mixture_ft_*.log"
  echo "Results: results_dir/sim_mixture/dim${DIM}/finetune/"
fi
