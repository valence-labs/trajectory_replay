#!/usr/bin/env bash
# Submit held-out (c,p) comparison sweep: noise-to-data vs both data-to-data variants.
#
#   ./sbatch/sim/submit_sweep_holdout_cmp.sh
#   DRY_RUN=1 ./sbatch/sim/submit_sweep_holdout_cmp.sh
#   MAX_CONCURRENT=10 ./sbatch/sim/submit_sweep_holdout_cmp.sh
#
# Grid: difficulty (simple, medium, hard) × coupling × HELD_OUT_CONTEXTS.
# Simulate N_CONTEXTS=10 (c = 0..9); default hold out 5 contexts: 2,4,6,8,9; HELD_OUT_PERT=3.
# EPOCHS=1000 → 3 × 3 × 5 = 45 jobs. Each job: TRAIN_BOTH_D2D_VARIANTS=1 (all three methods).
#
# Difficulty (context_scale / pert_scale / interaction_scale for hadamard):
#   simple — well-separated strata, weak interaction (easy OOD)
#   medium — moderate separation and hadamard strength
#   hard   — large context/pert shifts, strong interaction
#
# Results: ./results_dir/simulation_results/sweep_holdout_cmp/<regime>_coup_<coupling>_hc<c>/
# Plots:    ./sbatch/sim/submit_plot_holdout_cmp_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

RESULTS_DIR="./results_dir"
mkdir -p "${SBATCH_ROOT}/logs"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_combo_holdout_compare.sbatch"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_DELAY_SEC="${SUBMIT_DELAY_SEC:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
SWEEP_JOB_NAME="${SWEEP_JOB_NAME:-holdout_cmp_sweep}"

export N_CONTEXTS="${N_CONTEXTS:-10}"
export HELD_OUT_CONTEXTS="${HELD_OUT_CONTEXTS:-2,4,6,8,9}"
export HELD_OUT_PERT="${HELD_OUT_PERT:-3}"
export CONTROL_PERT="${CONTROL_PERT:-0}"
export EPOCHS="${EPOCHS:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-3}"
export TRAIN_SEED="${TRAIN_SEED:-0}"
export DEVICE="${DEVICE:-cpu}"
export TRAIN_BOTH_D2D_VARIANTS="${TRAIN_BOTH_D2D_VARIANTS:-1}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"

slurm_time_for_epochs() {
  case "$1" in
    200) echo "04:00:00" ;;
    500) echo "08:00:00" ;;
    1000) echo "12:00:00" ;;  # 3 models × 1000 epochs
    2000) echo "24:00:00" ;;
    *) echo "12:00:00" ;;
  esac
}

apply_regime() {
  case "$1" in
    simple)
      export CONTEXT_SCALE="${CONTEXT_SCALE_SIMPLE:-1.2}"
      export PERT_SCALE="${PERT_SCALE_SIMPLE:-0.9}"
      export INTERACTION_SCALE="${INTERACTION_SCALE_SIMPLE:-0.35}"
      ;;
    medium)
      export CONTEXT_SCALE="${CONTEXT_SCALE_MEDIUM:-2.0}"
      export PERT_SCALE="${PERT_SCALE_MEDIUM:-0.85}"
      export INTERACTION_SCALE="${INTERACTION_SCALE_MEDIUM:-0.55}"
      ;;
    hard)
      export CONTEXT_SCALE="${CONTEXT_SCALE_HARD:-4.0}"
      export PERT_SCALE="${PERT_SCALE_HARD:-1.0}"
      export INTERACTION_SCALE="${INTERACTION_SCALE_HARD:-1.0}"
      ;;
    *)
      echo "unknown regime: $1 (use simple, medium, or hard)" >&2
      return 1
      ;;
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

n_submitted=0
n_skipped=0
walltime="$(slurm_time_for_epochs "${EPOCHS}")"
last_context=$((N_CONTEXTS - 1))

_held_norm="${HELD_OUT_CONTEXTS//,/ }"
read -ra holdout_context_list <<< "${_held_norm}"

if [[ ${#holdout_context_list[@]} -eq 0 ]]; then
  echo "ERROR: HELD_OUT_CONTEXTS is empty" >&2
  exit 1
fi
for holdout_context in "${holdout_context_list[@]}"; do
  if [[ ! "${holdout_context}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid held-out context index: ${holdout_context}" >&2
    exit 1
  fi
  if (( holdout_context < 0 || holdout_context > last_context )); then
    echo "ERROR: held-out context ${holdout_context} out of range [0, ${last_context}] (N_CONTEXTS=${N_CONTEXTS})" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" != "1" && "${MAX_CONCURRENT}" -gt 0 ]]; then
  echo "Throttling: max ${MAX_CONCURRENT} concurrent jobs named ${SWEEP_JOB_NAME}"
fi
echo "Sweep: results_dir/simulation_results/sweep_holdout_cmp  epochs=${EPOCHS}  walltime=${walltime}"
echo "  regimes=simple,medium,hard  couplings=additive,rotated,hadamard"
echo "  simulate N_CONTEXTS=${N_CONTEXTS} (c=0..${last_context}); hold out ${#holdout_context_list[@]} contexts: ${holdout_context_list[*]}; p=${HELD_OUT_PERT}"

for regime in simple medium hard; do
  apply_regime "${regime}"
  for coupling in additive rotated hadamard; do
    for holdout_context in "${holdout_context_list[@]}"; do
      run_name="simulation_results/sweep_holdout_cmp/${regime}_coup_${coupling}_hc${holdout_context}"
      log_file="${SBATCH_ROOT}/logs/sweep_holdout_cmp_${regime}_coup_${coupling}_hc${holdout_context}_%j.log"

      if [[ -f "${RESULTS_DIR}/${run_name}/metrics.json" ]]; then
        echo "skip (exists): ${run_name}"
        n_skipped=$((n_skipped + 1))
        continue
      fi

      export_cmd="ALL,SIM_ROOT=${SIM_ROOT},RUN_NAME=${run_name}"
      export_cmd+=",COUPLING=${coupling}"
      export_cmd+=",N_CONTEXTS=${N_CONTEXTS}"
      export_cmd+=",EPOCHS=${EPOCHS}"
      export_cmd+=",TRAIN_BOTH_D2D_VARIANTS=${TRAIN_BOTH_D2D_VARIANTS}"
      export_cmd+=",SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS}"
      export_cmd+=",CONTEXT_SCALE=${CONTEXT_SCALE}"
      export_cmd+=",PERT_SCALE=${PERT_SCALE}"
      export_cmd+=",INTERACTION_SCALE=${INTERACTION_SCALE}"
      export_cmd+=",HELD_OUT_CONTEXT=${holdout_context}"
      export_cmd+=",HELD_OUT_PERT=${HELD_OUT_PERT}"
      export_cmd+=",CONTROL_PERT=${CONTROL_PERT}"
      export_cmd+=",BATCH_SIZE=${BATCH_SIZE}"
      export_cmd+=",LR=${LR}"
      export_cmd+=",TRAIN_SEED=${TRAIN_SEED}"
      export_cmd+=",DEVICE=${DEVICE}"
      export_cmd+=",SWEEP_REGIME=${regime}"

      sbatch_cmd=(
        sbatch
        --export="${export_cmd}"
        --job-name="${SWEEP_JOB_NAME}"
        --time="${walltime}"
        --output="${log_file}"
        "${SBATCH_SCRIPT}"
      )

      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "${sbatch_cmd[*]}"
        echo "  regime=${regime} coupling=${coupling} hold_c=${holdout_context} ctx=${CONTEXT_SCALE} pert=${PERT_SCALE} int=${INTERACTION_SCALE}"
      else
        if [[ "${MAX_CONCURRENT}" -gt 0 ]]; then
          wait_for_slot
        fi
        echo "submit: ${run_name}  (${regime}, ${coupling}, hold_c=${holdout_context})"
        "${sbatch_cmd[@]}"
        n_submitted=$((n_submitted + 1))
        if [[ "${SUBMIT_DELAY_SEC}" -gt 0 ]]; then
          sleep "${SUBMIT_DELAY_SEC}"
        fi
      fi
    done
  done
done

echo "Done. submitted=${n_submitted} skipped=${n_skipped} dry_run=${DRY_RUN}"
echo "After jobs finish, plot with:"
echo "  ./sbatch/sim/submit_plot_holdout_cmp_sweep.sh"
