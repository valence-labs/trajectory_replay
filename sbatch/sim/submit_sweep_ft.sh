#!/usr/bin/env bash
# Submit finetuning hyperparameter sweep via Slurm.
#
#   ./sbatch/sim/submit_sweep_ft.sh              # submit all jobs
#   DRY_RUN=1 ./sbatch/sim/submit_sweep_ft.sh    # print sbatch commands only
#   MAX_CONCURRENT=10 ./sbatch/sim/submit_sweep_ft.sh   # at most 10 sweep jobs in queue (default)
#   MAX_CONCURRENT=0 ./sbatch/sim/submit_sweep_ft.sh    # no throttle (submit all at once)
#
# Results: ./results_dir/simulation_results/sweep_ft_small_pert_effect/<name>/
# Logs:    sbatch/logs/sweep_ft_<run_name>_<jobid>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

RESULTS_DIR="./results_dir"
mkdir -p "${SBATCH_ROOT}/logs"

SBATCH_SCRIPT="${SCRIPT_DIR}/train_perturbation_forgetting.sbatch"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_DELAY_SEC="${SUBMIT_DELAY_SEC:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
SWEEP_JOB_NAME="${SWEEP_JOB_NAME:-ft_sweep}"

# Shared simulation / training defaults (override via env when calling this script).
export HELD_OUT_PERT="${HELD_OUT_PERT:-3}"
export HELD_OUT_CONTEXT="${HELD_OUT_CONTEXT:-2}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
export FINETUNE_LR="${FINETUNE_LR:-1e-3}"
export TRAIN_SEED="${TRAIN_SEED:-0}"
export DEVICE="${DEVICE:-cpu}"

bool_to_flag() {
  case "${1,,}" in
    true | 1 | yes) echo "1" ;;
    false | 0 | no) echo "0" ;;
    *) echo "invalid bool: $1" >&2; return 1 ;;
  esac
}

holdout_tag() {
  case "$1" in
    context) echo "hold_ctx" ;;
    perturbation) echo "hold_pert" ;;
    combo) echo "hold_combo" ;;
    *) echo "$1" ;;
  esac
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

n_submitted=0
n_skipped=0

if [[ "${DRY_RUN}" != "1" && "${MAX_CONCURRENT}" -gt 0 ]]; then
  echo "Throttling: max ${MAX_CONCURRENT} concurrent jobs named ${SWEEP_JOB_NAME}"
fi

for holdout in context perturbation combo; do
  for coupling in additive rotated hadamard; do
    for ftep in 100000; do  # 1000 10000 50000 
      for train_vf in True False; do
        for train_emb in True False; do
          for ft_data in held_out; do  # held_out all
            vf_flag="$(bool_to_flag "${train_vf}")"
            emb_flag="$(bool_to_flag "${train_emb}")"
            hold_tag="$(holdout_tag "${holdout}")"

            run_name="simulation_results/sweep_ft_small_pert_effect/${hold_tag}_coup_${coupling}_ftep_${ftep}_vf_${vf_flag}_emb_${emb_flag}_data_${ft_data}"
            log_file="${SBATCH_ROOT}/logs/${run_name//\//_}_%j.log"
            walltime="$(slurm_time_for_epochs "${ftep}")"

            if [[ -d "${RESULTS_DIR}/${run_name}" && -f "${RESULTS_DIR}/${run_name}/metrics.json" ]]; then
              echo "skip (exists): ${run_name}"
              n_skipped=$((n_skipped + 1))
              continue
            fi

            export_cmd="ALL,SIM_ROOT=${SIM_ROOT},RUN_NAME=${run_name}"
            export_cmd+=",HOLDOUT_MODE=${holdout}"
            export_cmd+=",COUPLING=${coupling}"
            export_cmd+=",FINETUNE_EPOCHS=${ftep}"
            export_cmd+=",FINETUNE_TRAIN_VF=${vf_flag}"
            export_cmd+=",FINETUNE_TRAIN_OLD_EMBEDDINGS=${emb_flag}"
            export_cmd+=",FINETUNE_DATA=${ft_data}"

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
            else
              if [[ "${MAX_CONCURRENT}" -gt 0 ]]; then
                wait_for_slot
              fi
              echo "submit: ${run_name}  (time=${walltime})"
              "${sbatch_cmd[@]}"
              n_submitted=$((n_submitted + 1))
              if [[ "${SUBMIT_DELAY_SEC}" -gt 0 ]]; then
                sleep "${SUBMIT_DELAY_SEC}"
              fi
            fi
          done
        done
      done
    done
  done
done

echo "Done. submitted=${n_submitted} skipped=${n_skipped} dry_run=${DRY_RUN}"

# Replay example (fraction in every batch; optional tag replay_r25 in run name):
#   for ft_data in held_out replay; do
#     [[ "${ft_data}" == replay ]] && ft_data="replay_r25"
#     export FINETUNE_DATA=replay REPLAY_FRACTION=0.25 REPLAY_MAX_SAMPLES=600
# for ft_data in held_out all; do