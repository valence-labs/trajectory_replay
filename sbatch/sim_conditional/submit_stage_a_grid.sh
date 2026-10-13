#!/usr/bin/env bash
# Submit Stage A pretrain, then λ-sweep FT (naive + gen + traj).
#
#   ./sbatch/sim_conditional/submit_stage_a_grid.sh
#   DRY_RUN=1 ./sbatch/sim_conditional/submit_stage_a_grid.sh
#   SEEDS="0 1 2" LAMBDAS="0.1 0.5 0.9" ./sbatch/sim_conditional/submit_stage_a_grid.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
# Prefer STAGE_A_DIM so a leftover DIM=256 from sim_mixture does not leak in.
DIM="${STAGE_A_DIM:-2}"
LAMBDAS=(${LAMBDAS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9})
SEEDS=(${SEEDS:-0})
STRATEGIES=(${STRATEGIES:-naive generative trajectory})
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
FT_EPOCHS="${FT_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
TRAJ_KAPPA="${TRAJ_KAPPA:-none}"

export SIM_ROOT DIM BATCH_SIZE LR DEVICE TRAJ_KAPPA

echo "=== Stage A grid ==="
echo "DIM=${DIM} SEEDS=${SEEDS[*]} LAMBDAS=${LAMBDAS[*]} STRATEGIES=${STRATEGIES[*]}"

submit_pretrain() {
  local seed="$1"
  export TRAIN_SEED="${seed}"
  export EPOCHS="${PRETRAIN_EPOCHS}"
  export RUN_NAME="pretrain_seed${seed}"
  local ckpt="${SIM_ROOT}/results_dir/sim_conditional/stage_a_leaveout/dim${DIM}/${RUN_NAME}/checkpoint.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "Reuse pretrain seed=${seed}: ${ckpt}" >&2
    echo ""
    return 0
  fi
  local log_file="${SBATCH_ROOT}/logs/sim_conditional_a_pre_s${seed}_%j.log"
  local cmd=(
    sbatch --export=ALL --parsable
    --job-name="simcond-a-pre-s${seed}"
    --output="${log_file}"
    "${SCRIPT_DIR}/train_pretrain.sbatch"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY ${cmd[*]}" >&2
    echo "dry-pre-s${seed}"
    return 0
  fi
  local jid
  jid="$("${cmd[@]}")"
  echo "Submitted pretrain seed=${seed} job=${jid}" >&2
  echo "${jid}"
}

submit_ft() {
  local strategy="$1"
  local lam="$2"
  local seed="$3"
  local dep="${4:-}"
  export STRATEGY="${strategy}"
  export LAMBDA_MIX="${lam}"
  export TRAIN_SEED="${seed}"
  export EPOCHS="${FT_EPOCHS}"
  export PRETRAIN_CHECKPOINT="${SIM_ROOT}/results_dir/sim_conditional/stage_a_leaveout/dim${DIM}/pretrain_seed${seed}/checkpoint.pt"
  if [[ "${strategy}" == "naive" ]]; then
    export RUN_NAME="ft_naive_e${FT_EPOCHS}_seed${seed}"
  else
    export RUN_NAME="ft_${strategy}_lam${lam}_e${FT_EPOCHS}_seed${seed}"
  fi
  local log_file="${SBATCH_ROOT}/logs/sim_conditional_a_${RUN_NAME//./p}_%j.log"
  local cmd=(
    sbatch --export=ALL --parsable
    --job-name="simcond-a-${strategy:0:4}-s${seed}"
    --output="${log_file}"
  )
  if [[ -n "${dep}" && "${dep}" != dry-* && "${DRY_RUN}" != "1" ]]; then
    cmd+=(--dependency="afterok:${dep}")
  fi
  cmd+=("${SCRIPT_DIR}/train_finetune.sbatch")
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY dep=${dep:-none} ${cmd[*]}  # ${RUN_NAME}" >&2
    return 0
  fi
  local jid
  jid="$("${cmd[@]}")"
  echo "Submitted ${RUN_NAME} job=${jid} dep=${dep:-none}" >&2
}

for seed in "${SEEDS[@]}"; do
  pre_jid="$(submit_pretrain "${seed}")"
  for strategy in "${STRATEGIES[@]}"; do
    if [[ "${strategy}" == "naive" ]]; then
      submit_ft "naive" "0.0" "${seed}" "${pre_jid}"
    else
      for lam in "${LAMBDAS[@]}"; do
        submit_ft "${strategy}" "${lam}" "${seed}" "${pre_jid}"
      done
    fi
  done
done

echo "Done submitting Stage A grid."
