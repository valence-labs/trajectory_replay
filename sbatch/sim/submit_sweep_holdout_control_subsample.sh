#!/usr/bin/env bash
# Held-out (c,p) sweep vs control pool size (control rows dropped from dataset).
#
# Same grid as submit_sweep_holdout_cmp.sh (simple/medium/hard × coupling × held-out c),
# plus control subsample fractions. TRAIN_BOTH_D2D_VARIANTS=1 (noise + two d2d) per job.
#
#   ./sbatch/sim/submit_sweep_holdout_control_subsample.sh
#   DRY_RUN=1 ./sbatch/sim/submit_sweep_holdout_control_subsample.sh
#
# Default: 3 × 3 × 5 × 5 = 225 jobs (fracs: 1.0 0.5 0.25 0.1 0.05), EPOCHS=1000.
# Results: results_dir/simulation_results/sweep_holdout_ctrl/<regime>_coup_<coupling>_hc<c>_cf<frac>/
# Plot:    ./sbatch/sim/submit_plot_holdout_cmp_sweep.sh results_dir/simulation_results/sweep_holdout_ctrl ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

RESULTS_DIR="./results_dir"
mkdir -p "${SBATCH_ROOT}/logs"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_combo_holdout_compare.sbatch"
DRY_RUN="${DRY_RUN:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
SWEEP_JOB_NAME="${SWEEP_JOB_NAME:-holdout_ctrl_sweep}"

export N_CONTEXTS="${N_CONTEXTS:-10}"
export HELD_OUT_CONTEXTS="${HELD_OUT_CONTEXTS:-2,6,9}"
export HELD_OUT_PERT="${HELD_OUT_PERT:-3}"
export CONTROL_PERT="${CONTROL_PERT:-0}"
export EPOCHS="${EPOCHS:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-3}"
export TRAIN_SEED="${TRAIN_SEED:-0}"
export DEVICE="${DEVICE:-cpu}"
export TRAIN_BOTH_D2D_VARIANTS="${TRAIN_BOTH_D2D_VARIANTS:-1}"
export D2D_EXCLUDE_IDENTITY_TARGETS="${D2D_EXCLUDE_IDENTITY_TARGETS:-0}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"

# Fractions of control cells kept per context (same count across contexts).
CONTROL_FRACS="${CONTROL_FRACS:-0.8 0.5 0.1 0.01}"

slurm_time_for_epochs() {
  case "$1" in
    200) echo "04:00:00" ;;
    500) echo "08:00:00" ;;
    1000) echo "12:00:00" ;;
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
      export CONTEXT_SCALE="${CONTEXT_SCALE_MEDIUM:-2.5}"
      export PERT_SCALE="${PERT_SCALE_MEDIUM:-0.85}"
      export INTERACTION_SCALE="${INTERACTION_SCALE_MEDIUM:-0.55}"
      ;;
    hard)
      export CONTEXT_SCALE="${CONTEXT_SCALE_HARD:-5.0}"
      export PERT_SCALE="${PERT_SCALE_HARD:-1.0}"
      export INTERACTION_SCALE="${INTERACTION_SCALE_HARD:-1.0}"
      ;;
    *)
      echo "unknown regime: $1" >&2
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
    echo "  throttle: ${n}/${MAX_CONCURRENT} active — waiting ${POLL_INTERVAL_SEC}s"
    sleep "${POLL_INTERVAL_SEC}"
    n="$(count_active_sweep_jobs)"
  done
}

walltime="$(slurm_time_for_epochs "${EPOCHS}")"
last_context=$((N_CONTEXTS - 1))

_held_norm="${HELD_OUT_CONTEXTS//,/ }"
read -ra holdout_context_list <<< "${_held_norm}"

for holdout_context in "${holdout_context_list[@]}"; do
  if [[ ! "${holdout_context}" =~ ^[0-9]+$ ]] || (( holdout_context < 0 || holdout_context > last_context )); then
    echo "ERROR: invalid held-out context: ${holdout_context}" >&2
    exit 1
  fi
done

n_submitted=0
n_skipped=0

echo "Control subsample sweep: sweep_holdout_ctrl  epochs=${EPOCHS}  fracs=${CONTROL_FRACS}"

for ctrl_frac in ${CONTROL_FRACS}; do
  export CONTROL_SUBSAMPLE_FRAC="${ctrl_frac}"
  frac_tag=$(printf '%s' "${ctrl_frac}" | tr '.' 'p')

  for regime in simple medium hard; do
    apply_regime "${regime}"
    for coupling in additive rotated hadamard; do
      for holdout_context in "${holdout_context_list[@]}"; do
        run_name="simulation_results/sweep_holdout_ctrl/${regime}_coup_${coupling}_hc${holdout_context}_cf${frac_tag}"
        log_file="${SBATCH_ROOT}/logs/sweep_holdout_ctrl_${regime}_${coupling}_hc${holdout_context}_cf${frac_tag}_%j.log"

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
        export_cmd+=",D2D_EXCLUDE_IDENTITY_TARGETS=${D2D_EXCLUDE_IDENTITY_TARGETS}"
        export_cmd+=",SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS}"
        export_cmd+=",CONTROL_SUBSAMPLE_FRAC=${CONTROL_SUBSAMPLE_FRAC}"
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
          echo "  frac=${ctrl_frac} regime=${regime} ${coupling} hold_c=${holdout_context}"
        else
          if [[ "${MAX_CONCURRENT}" -gt 0 ]]; then
            wait_for_slot
          fi
          echo "submit: ${run_name}"
          "${sbatch_cmd[@]}"
          n_submitted=$((n_submitted + 1))
        fi
      done
    done
  done
done

echo "Done. submitted=${n_submitted} skipped=${n_skipped} dry_run=${DRY_RUN}"
echo "Plot when done:"
echo "  ./sbatch/sim/submit_plot_holdout_cmp_sweep.sh results_dir/simulation_results/sweep_holdout_ctrl results_dir/simulation_results/sweep_holdout_ctrl/metrics_plots"
