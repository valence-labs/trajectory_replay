#!/usr/bin/env bash
# Stage A λ-grid with batch-of-N submission (default N=10).
# Submits up to JOB_BATCH jobs, waits until that batch finishes, then next batch.
#
#   ./sbatch/sim_conditional/submit_stage_a_uncond_throttled.sh
#   DRY_RUN=1 ./sbatch/sim_conditional/submit_stage_a_uncond_throttled.sh
#   JOB_BATCH=10 SEEDS="0 1 2" LAMBDAS="0.1 0.25 0.5 0.75 0.9" \
#     ./sbatch/sim_conditional/submit_stage_a_uncond_throttled.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
# Max jobs submitted per wave (wait for the wave to finish before the next).
JOB_BATCH="${JOB_BATCH:-10}"
POLL_SEC="${POLL_SEC:-60}"
DIM="${STAGE_A_DIM:-2}"
SEEDS=(${SEEDS:-0 1 2})
LAMBDAS=(${LAMBDAS:-0.1 0.25 0.5 0.75 0.9})
STRATEGIES=(${STRATEGIES:-naive generative trajectory})
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
FT_EPOCHS="${FT_EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
TRAJ_KAPPA="${TRAJ_KAPPA:-none}"

export SIM_ROOT DIM LR DEVICE TRAJ_KAPPA

echo "=== Stage A uncond throttled grid ==="
echo "DIM=${DIM} SEEDS=${SEEDS[*]} LAMBDAS=${LAMBDAS[*]}"
echo "JOB_BATCH=${JOB_BATCH} POLL_SEC=${POLL_SEC} FT_EPOCHS=${FT_EPOCHS}"

wait_for_jobs() {
  local -a jids=("$@")
  if ((${#jids[@]} == 0)); then
    return 0
  fi
  local csv
  csv="$(IFS=,; echo "${jids[*]}")"
  echo "Waiting for batch jobs: ${csv}"
  while true; do
    local active
    active="$(squeue -j "${csv}" -h -o '%i' 2>/dev/null | wc -l | tr -d ' ')"
    active="${active:-0}"
    if (( active == 0 )); then
      echo "Batch finished."
      return 0
    fi
    echo "  batch still active=${active}/${#jids[@]}; sleeping ${POLL_SEC}s"
    sleep "${POLL_SEC}"
  done
}

# Status → stderr; job id (or empty) → stdout.
submit_pretrain() {
  local seed="$1"
  export TRAIN_SEED="${seed}"
  export EPOCHS="${PRETRAIN_EPOCHS}"
  export RUN_NAME="pretrain_seed${seed}"
  local ckpt="${SIM_ROOT}/results_dir/sim_conditional/stage_a_uncond_leaveout/dim${DIM}/${RUN_NAME}/checkpoint.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "Reuse pretrain seed=${seed}" >&2
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY pretrain seed=${seed}" >&2
    echo "dry-pre-s${seed}"
    return 0
  fi
  local log_file="${SBATCH_ROOT}/logs/sim_conditional_au_pre_s${seed}_%j.log"
  local jid
  jid="$(sbatch --export=ALL --parsable \
    --job-name="simcond-au-pre-s${seed}" \
    --output="${log_file}" \
    "${SCRIPT_DIR}/train_pretrain_uncond.sbatch")"
  echo "Submitted pretrain seed=${seed} job=${jid}" >&2
  echo "${jid}"
}

# --- Pretrains first ---
PRE_JIDS=()
for seed in "${SEEDS[@]}"; do
  jid="$(submit_pretrain "${seed}" || true)"
  if [[ -n "${jid}" ]]; then
    PRE_JIDS+=("${jid}")
  fi
done
if [[ "${DRY_RUN}" != "1" && ${#PRE_JIDS[@]} -gt 0 ]]; then
  wait_for_jobs "${PRE_JIDS[@]}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    ckpt="${SIM_ROOT}/results_dir/sim_conditional/stage_a_uncond_leaveout/dim${DIM}/pretrain_seed${seed}/checkpoint.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "ERROR: missing pretrain checkpoint after wait: ${ckpt}" >&2
      exit 1
    fi
  done
fi

# --- Build FT work list; skip completed runs ---
WORK=()
for seed in "${SEEDS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    if [[ "${strategy}" == "naive" ]]; then
      run="ft_naive_e${FT_EPOCHS}_seed${seed}"
      metrics="${SIM_ROOT}/results_dir/sim_conditional/stage_a_uncond_leaveout/dim${DIM}/finetune/${run}/metrics.json"
      if [[ -f "${metrics}" ]]; then
        echo "Skip completed ${run}"
      else
        WORK+=("naive|-|${seed}")
      fi
    else
      for lam in "${LAMBDAS[@]}"; do
        run="ft_${strategy}_lam${lam}_e${FT_EPOCHS}_seed${seed}"
        metrics="${SIM_ROOT}/results_dir/sim_conditional/stage_a_uncond_leaveout/dim${DIM}/finetune/${run}/metrics.json"
        if [[ -f "${metrics}" ]]; then
          echo "Skip completed ${run}"
        else
          WORK+=("${strategy}|${lam}|${seed}")
        fi
      done
    fi
  done
done

n="${#WORK[@]}"
echo "FT jobs to submit: ${n} (job batch ${JOB_BATCH})"

submit_ft() {
  local strategy="$1"
  local lam="$2"
  local seed="$3"
  export STRATEGY="${strategy}"
  export TRAIN_SEED="${seed}"
  export EPOCHS="${FT_EPOCHS}"
  export PRETRAIN_CHECKPOINT="${SIM_ROOT}/results_dir/sim_conditional/stage_a_uncond_leaveout/dim${DIM}/pretrain_seed${seed}/checkpoint.pt"
  if [[ "${strategy}" == "naive" ]]; then
    export LAMBDA_MIX=0.5
    export RUN_NAME="ft_naive_e${FT_EPOCHS}_seed${seed}"
  else
    export LAMBDA_MIX="${lam}"
    export RUN_NAME="ft_${strategy}_lam${lam}_e${FT_EPOCHS}_seed${seed}"
  fi
  local log_file="${SBATCH_ROOT}/logs/sim_conditional_au_${RUN_NAME//./p}_%j.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY ${RUN_NAME}" >&2
    echo "dry-${RUN_NAME}"
    return 0
  fi
  local jid
  jid="$(sbatch --export=ALL --parsable \
    --job-name="simcond-au-${strategy:0:4}-s${seed}" \
    --output="${log_file}" \
    "${SCRIPT_DIR}/train_finetune_uncond.sbatch")"
  echo "Submitted ${RUN_NAME} job=${jid}" >&2
  echo "${jid}"
}

idx=0
batch_num=0
while (( idx < n )); do
  batch_num=$((batch_num + 1))
  BATCH_JIDS=()
  batch_end=$((idx + JOB_BATCH))
  if (( batch_end > n )); then
    batch_end=${n}
  fi
  echo "=== Batch ${batch_num}: jobs $((idx + 1))–${batch_end} of ${n} ==="
  while (( idx < batch_end )); do
    IFS='|' read -r strategy lam seed <<< "${WORK[$idx]}"
    jid="$(submit_ft "${strategy}" "${lam}" "${seed}")"
    if [[ -n "${jid}" ]]; then
      BATCH_JIDS+=("${jid}")
    fi
    idx=$((idx + 1))
  done
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY: would wait for ${#BATCH_JIDS[@]} jobs"
    continue
  fi
  wait_for_jobs "${BATCH_JIDS[@]}"
done

echo "=== All Stage A jobs submitted/finished ==="
done_n="$(find "results_dir/sim_conditional/stage_a_uncond_leaveout/dim${DIM}/finetune" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')"
echo "metrics.json count under finetune/: ${done_n}"
