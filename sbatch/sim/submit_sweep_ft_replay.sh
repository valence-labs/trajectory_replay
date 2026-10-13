#!/usr/bin/env bash
# Submit replay-buffer finetune sweep (subset of vf/emb × holdout configs).
#
#   ./sbatch/sim/submit_sweep_ft_replay.sh
#   DRY_RUN=1 ./sbatch/sim/submit_sweep_ft_replay.sh
#   MAX_CONCURRENT=10 ./sbatch/sim/submit_sweep_ft_replay.sh
#
# Config grid (FINETUNE_DATA=replay; OOD = held-out strata only):
#   vf+ emb+  →  holdout: context, perturbation, combo
#   vf- emb-  →  holdout: combo only (held-out c and p embedding rows)
#
# Swept: coupling × replay_fraction  (finetune epochs fixed at 10k)
#
# Results: ./results_dir/${SWEEP_ROOT}/<run_name>/
# Logs:     sbatch/logs/${run_name//\//_}_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

RESULTS_DIR="./results_dir"
SWEEP_ROOT="${SWEEP_ROOT:-simulation_results/sweep_ft_replay}"
mkdir -p "${SBATCH_ROOT}/logs"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_perturbation_forgetting.sbatch"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_DELAY_SEC="${SUBMIT_DELAY_SEC:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
SWEEP_JOB_NAME="${SWEEP_JOB_NAME:-ft_replay_sweep}"

# Fixed finetune length (user request).
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10000}"

# Replay sweep axes (space-separated).
REPLAY_FRACTIONS="${REPLAY_FRACTIONS:-0.001 0.01 0.1 0.25 0.5}"
COUPLINGS="${COUPLINGS:-additive rotated hadamard}"

# Shared training defaults (match submit_sweep_ft.sh / small_pert_effect).
export HELD_OUT_PERT="${HELD_OUT_PERT:-3}"
export HELD_OUT_CONTEXT="${HELD_OUT_CONTEXT:-2}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
export FINETUNE_LR="${FINETUNE_LR:-1e-3}"
export TRAIN_SEED="${TRAIN_SEED:-0}"
export DEVICE="${DEVICE:-cpu}"
export REPLAY_MAX_SAMPLES="${REPLAY_MAX_SAMPLES:-600}"

holdout_tag() {
  case "$1" in
    context) echo "hold_ctx" ;;
    perturbation) echo "hold_pert" ;;
    combo) echo "hold_combo" ;;
    *) echo "$1" ;;
  esac
}

# 0.25 → replay_r25
replay_data_tag() {
  local frac="$1"
  local pct
  pct="$(awk -v f="${frac}" 'BEGIN { printf "%d", f * 100 }')"
  echo "replay_r${pct}"
}

slurm_time_for_epochs() {
  case "$1" in
    200) echo "04:00:00" ;;
    1000) echo "08:00:00" ;;
    10000) echo "24:00:00" ;;
    *) echo "12:00:00" ;;
  esac
}

count_active_sweep_jobs() {
  if ! command -v squeue &>/dev/null; then
    echo 0
    return
  fi
  squeue -h -u "${USER}" -n "${SWEEP_JOB_NAME}" -t RUNNING,PENDING,CONFIGURING 2>/dev/null | wc -l | tr -d ' '
}

wait_for_slot() {
  local n
  n="$(count_active_sweep_jobs)"
  while [[ "${n}" -ge "${MAX_CONCURRENT}" ]]; do
    echo "  throttle: ${n}/${MAX_CONCURRENT} jobs (${SWEEP_JOB_NAME}) active — waiting ${POLL_INTERVAL_SEC}s"
    sleep "${POLL_INTERVAL_SEC}"
    n="$(count_active_sweep_jobs)"
  done
}

submit_replay_job() {
  local holdout="$1"
  local train_vf="$2"
  local train_emb="$3"
  local coupling="$4"
  local replay_frac="$5"

  local hold_tag data_tag run_name log_file walltime export_cmd
  hold_tag="$(holdout_tag "${holdout}")"
  data_tag="$(replay_data_tag "${replay_frac}")"
  run_name="${SWEEP_ROOT}/${hold_tag}_coup_${coupling}_ftep_${FINETUNE_EPOCHS}_vf_${train_vf}_emb_${train_emb}_data_${data_tag}"
  log_file="${SBATCH_ROOT}/logs/${run_name//\//_}_%j.log"
  walltime="$(slurm_time_for_epochs "${FINETUNE_EPOCHS}")"

  if [[ -d "${RESULTS_DIR}/${run_name}" && -f "${RESULTS_DIR}/${run_name}/metrics.json" ]]; then
    echo "skip (exists): ${run_name}"
    n_skipped=$((n_skipped + 1))
    return
  fi

  export_cmd="ALL,SIM_ROOT=${SIM_ROOT},RUN_NAME=${run_name}"
  export_cmd+=",HOLDOUT_MODE=${holdout}"
  export_cmd+=",COUPLING=${coupling}"
  export_cmd+=",FINETUNE_EPOCHS=${FINETUNE_EPOCHS}"
  export_cmd+=",FINETUNE_TRAIN_VF=${train_vf}"
  export_cmd+=",FINETUNE_TRAIN_OLD_EMBEDDINGS=${train_emb}"
  export_cmd+=",FINETUNE_DATA=replay"
  export_cmd+=",REPLAY_FRACTION=${replay_frac}"
  if [[ -n "${REPLAY_MAX_SAMPLES}" ]]; then
    export_cmd+=",REPLAY_MAX_SAMPLES=${REPLAY_MAX_SAMPLES}"
  fi

  local -a sbatch_cmd=(
    sbatch
    --export="${export_cmd}"
    --job-name="${SWEEP_JOB_NAME}"
    --time="${walltime}"
    --output="${log_file}"
    "${SBATCH_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${sbatch_cmd[*]}"
  else
    if [[ "${MAX_CONCURRENT}" -gt 0 ]]; then
      wait_for_slot
    fi
    echo "submit: ${run_name}  (replay=${replay_frac}, time=${walltime})"
    "${sbatch_cmd[@]}"
    n_submitted=$((n_submitted + 1))
    if [[ "${SUBMIT_DELAY_SEC}" -gt 0 ]]; then
      sleep "${SUBMIT_DELAY_SEC}"
    fi
  fi
}

n_submitted=0
n_skipped=0
walltime="$(slurm_time_for_epochs "${FINETUNE_EPOCHS}")"

echo "Replay sweep: ${SWEEP_ROOT}"
echo "  finetune_epochs=${FINETUNE_EPOCHS}  walltime=${walltime}"
echo "  replay_fractions=${REPLAY_FRACTIONS}"
echo "  couplings=${COUPLINGS}  replay_max_samples=${REPLAY_MAX_SAMPLES:-full pool}"

if [[ "${DRY_RUN}" != "1" && "${MAX_CONCURRENT}" -gt 0 ]]; then
  echo "Throttling: max ${MAX_CONCURRENT} concurrent jobs named ${SWEEP_JOB_NAME}"
fi

# vf+ | emb+ → all holdout types
for holdout in context perturbation combo; do
  for coupling in ${COUPLINGS}; do
    for replay_frac in ${REPLAY_FRACTIONS}; do
      submit_replay_job "${holdout}" 1 1 "${coupling}" "${replay_frac}"
    done
  done
done

# vf- | emb- → combo only (held-out c and p embedding rows)
for holdout in combo; do
  for coupling in ${COUPLINGS}; do
    for replay_frac in ${REPLAY_FRACTIONS}; do
      submit_replay_job "${holdout}" 0 0 "${coupling}" "${replay_frac}"
    done
  done
done

echo "Done. submitted=${n_submitted} skipped=${n_skipped} dry_run=${DRY_RUN}"
