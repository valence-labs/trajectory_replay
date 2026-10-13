#!/usr/bin/env bash
# Stage B λ-grid with batch-of-N submission (default N=10).
# Submits up to JOB_BATCH jobs, waits until that batch finishes, then next batch.
#
# Layout: results_dir/sim_conditional/stage_b_shift/dim{D}/c{C}_p{P}/
#
# Defaults reproduce the paper run: c6_p4, dim=2, seeds 10-50, bs=1024, lr=3e-4,
# pretrain 1000 ep, FT 2000 ep, no grad clipping.
#
# Run in tmux (recommended):
#   tmux new -s stage_b
#   ./sbatch/sim_conditional/submit_stage_b_throttled.sh
#
# Optional overrides:
#   DRY_RUN=1 ./sbatch/sim_conditional/submit_stage_b_throttled.sh
#   JOB_BATCH=10 SEEDS="10 20 30" N_CONTEXTS=10 \
#     LAMBDAS="0.05 0.1 0.25 0.5 0.75 0.9 0.95" \
#     FT_EPOCHS=2000 ./sbatch/sim_conditional/submit_stage_b_throttled.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"
mkdir -p "${SBATCH_ROOT}/logs"

DRY_RUN="${DRY_RUN:-0}"
JOB_BATCH="${JOB_BATCH:-10}"
POLL_SEC="${POLL_SEC:-60}"
DIM="${STAGE_B_DIM:-2}"
N_CONTEXTS="${N_CONTEXTS:-6}"
N_PERTURBATIONS="${N_PERTURBATIONS:-4}"
CONFIG_TAG="c${N_CONTEXTS}_p${N_PERTURBATIONS}"
DATA_ROOT="${SIM_ROOT}/results_dir/sim_conditional/stage_b_shift/data/dim${DIM}/${CONFIG_TAG}"
RUNS_ROOT="${SIM_ROOT}/results_dir/sim_conditional/stage_b_shift/dim${DIM}/${CONFIG_TAG}"
SEEDS=(${SEEDS:-10 20 30 40 50})
LAMBDAS=(${LAMBDAS:-0.05 0.1 0.25 0.5 0.75 0.9 0.95})
STRATEGIES=(${STRATEGIES:-naive generative trajectory})
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1000}"
FT_EPOCHS="${FT_EPOCHS:-2000}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-3e-4}"
DEVICE="${DEVICE:-cuda}"
TRAJ_KAPPA="${TRAJ_KAPPA:-none}"

export SIM_ROOT DIM N_CONTEXTS N_PERTURBATIONS LR DEVICE TRAJ_KAPPA

echo "=== Stage B shift throttled grid ==="
echo "DIM=${DIM} CONFIG=${CONFIG_TAG} SEEDS=${SEEDS[*]} LAMBDAS=${LAMBDAS[*]}"
echo "STRATEGIES=${STRATEGIES[*]}"
echo "JOB_BATCH=${JOB_BATCH} POLL_SEC=${POLL_SEC}"
echo "PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS} FT_EPOCHS=${FT_EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE} LR=${LR}"
echo "RUNS_ROOT=${RUNS_ROOT}"

if [[ ! -f "${DATA_ROOT}/meta.json" ]]; then
  echo "ERROR: missing Stage B data at ${DATA_ROOT}" >&2
  echo "Build it first:" >&2
  echo "  uv run python experiments/sim_conditional/stage_b_shift/build_datasets.py \\" >&2
  echo "    --n-contexts ${N_CONTEXTS} --n-perturbations ${N_PERTURBATIONS} --dim ${DIM}" >&2
  exit 1
fi

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
  local ckpt="${RUNS_ROOT}/${RUN_NAME}/checkpoint.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "Reuse pretrain seed=${seed}" >&2
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY pretrain seed=${seed}" >&2
    echo "dry-pre-s${seed}"
    return 0
  fi
  local log_file="${SBATCH_ROOT}/logs/sim_conditional_b_${CONFIG_TAG}_pre_s${seed}_%j.log"
  local jid
  jid="$(sbatch --export=ALL --parsable \
    --job-name="simcond-b-pre-c${N_CONTEXTS}-s${seed}" \
    --output="${log_file}" \
    "${SCRIPT_DIR}/train_pretrain_stage_b.sbatch")"
  echo "Submitted pretrain seed=${seed} job=${jid}" >&2
  echo "${jid}"
}

# --- Pretrains first (missing seeds only) ---
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
    ckpt="${RUNS_ROOT}/pretrain_seed${seed}/checkpoint.pt"
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
      metrics="${RUNS_ROOT}/finetune/${run}/metrics.json"
      if [[ -f "${metrics}" ]]; then
        echo "Skip completed ${run}"
      else
        WORK+=("naive|-|${seed}")
      fi
    else
      for lam in "${LAMBDAS[@]}"; do
        run="ft_${strategy}_lam${lam}_e${FT_EPOCHS}_seed${seed}"
        metrics="${RUNS_ROOT}/finetune/${run}/metrics.json"
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
  export PRETRAIN_CHECKPOINT="${RUNS_ROOT}/pretrain_seed${seed}/checkpoint.pt"
  if [[ "${strategy}" == "naive" ]]; then
    export LAMBDA_MIX=0.5
    export RUN_NAME="ft_naive_e${FT_EPOCHS}_seed${seed}"
  else
    export LAMBDA_MIX="${lam}"
    export RUN_NAME="ft_${strategy}_lam${lam}_e${FT_EPOCHS}_seed${seed}"
  fi
  local log_file="${SBATCH_ROOT}/logs/sim_conditional_b_${CONFIG_TAG}_${RUN_NAME//./p}_%j.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY ${RUN_NAME}" >&2
    echo "dry-${RUN_NAME}"
    return 0
  fi
  local jid
  jid="$(sbatch --export=ALL --parsable \
    --job-name="simcond-b-${strategy:0:4}-c${N_CONTEXTS}-s${seed}" \
    --output="${log_file}" \
    "${SCRIPT_DIR}/train_finetune_stage_b.sbatch")"
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

echo "=== All Stage B jobs submitted/finished ==="
done_n="$(find "${RUNS_ROOT}/finetune" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')"
echo "metrics.json count under ${CONFIG_TAG}/finetune/: ${done_n}"
