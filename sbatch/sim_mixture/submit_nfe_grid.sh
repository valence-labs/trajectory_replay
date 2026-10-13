#!/usr/bin/env bash
# Fresh NFE-curve grid: pretrain (seed0) then traj/gen FT with mid-training evals.
#
# Default protocol:
#   dims 2 16 64 128 256
#   strategies trajectory,generative
#   seeds 0 1 2
#   λ=0.5  lr=1e-3  κ=none
#   epochs=2564 (~100k steps)  eval_every=100
#
#   ./sbatch/sim_mixture/submit_nfe_grid.sh
#   DRY_RUN=1 ./sbatch/sim_mixture/submit_nfe_grid.sh
#   DIMS="64 256" SEEDS="0" ./sbatch/sim_mixture/submit_nfe_grid.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
DIMS=(${DIMS:-2 16 64 128 256})
STRATEGIES=(${STRATEGIES:-trajectory generative})
SEEDS=(${SEEDS:-0 1 2})

export SIM_ROOT
export LAMBDA_MIX="${LAMBDA_MIX:-0.5}"
export EPOCHS="${EPOCHS:-2564}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export N_GEN_EVAL="${N_GEN_EVAL:-4000}"
export N_GEN_EVAL_CURVE="${N_GEN_EVAL_CURVE:-2000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-3}"
export DEVICE="${DEVICE:-cuda}"
export ODE_METHOD="${ODE_METHOD:-euler}"
export ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.05}"
export TRAJ_KAPPA=none
# Allow override only via explicit NFE_TRAJ_KAPPA (avoids inheriting a stale TRAJ_KAPPA=ot).
if [[ -n "${NFE_TRAJ_KAPPA:-}" ]]; then
  export TRAJ_KAPPA="${NFE_TRAJ_KAPPA}"
fi
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"

echo "=== NFE grid submit ==="
echo "DIMS=${DIMS[*]}  STRATEGIES=${STRATEGIES[*]}  SEEDS=${SEEDS[*]}"
echo "EPOCHS=${EPOCHS}  EVAL_EVERY=${EVAL_EVERY}  LR=${LR}  TRAJ_KAPPA=${TRAJ_KAPPA}"

submit_pretrain() {
  local dim="$1"
  local ckpt="${SIM_ROOT}/results_dir/sim_mixture/dim${dim}/pretrain_task0_seed0/checkpoint.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "Reuse pretrain dim=${dim}: ${ckpt}" >&2
    echo ""
    return 0
  fi
  export DIM="${dim}"
  export TRAIN_SEED=0
  export EPOCHS="${PRETRAIN_EPOCHS}"
  export RUN_NAME="pretrain_task0_seed0"
  local log_file="${SBATCH_ROOT}/logs/sim_mixture_pretrain_dim${dim}_seed0_%j.log"
  local cmd=(
    sbatch --export=ALL --parsable
    --job-name="simmix-pre-d${dim}"
    --output="${log_file}"
    "${SCRIPT_DIR}/train_pretrain.sbatch"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY ${cmd[*]}" >&2
    echo "dry-pre-d${dim}"
    return 0
  fi
  local jid
  jid="$("${cmd[@]}")"
  echo "Submitted pretrain dim=${dim} job=${jid}" >&2
  echo "${jid}"
}

submit_ft() {
  local dim="$1"
  local strategy="$2"
  local seed="$3"
  local dep="${4:-}"
  export DIM="${dim}"
  export STRATEGY="${strategy}"
  export TRAIN_SEED="${seed}"
  export EPOCHS="${FT_EPOCHS}"
  export EVAL_EVERY="${FT_EVAL_EVERY}"
  export N_GEN_EVAL_CURVE
  export PRETRAIN_CHECKPOINT="${SIM_ROOT}/results_dir/sim_mixture/dim${dim}/pretrain_task0_seed0/checkpoint.pt"
  export RUN_NAME="ft_${strategy}_lam${LAMBDA_MIX}_e${FT_EPOCHS}_eval${FT_EVAL_EVERY}_seed${seed}"
  local log_file="${SBATCH_ROOT}/logs/sim_mixture_${RUN_NAME//./p}_%j.log"
  local cmd=(
    sbatch --export=ALL --parsable
    --job-name="simmix-${strategy:0:4}-d${dim}-s${seed}"
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

FT_EPOCHS="${EPOCHS}"
FT_EVAL_EVERY="${EVAL_EVERY}"

for dim in "${DIMS[@]}"; do
  pre_jid="$(submit_pretrain "${dim}")"
  for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      submit_ft "${dim}" "${strategy}" "${seed}" "${pre_jid}"
    done
  done
done

echo "Done submitting."
