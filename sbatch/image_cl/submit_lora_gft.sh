#!/usr/bin/env bash
# Submit the memory-free finetuning controls (LoRA and gradual finetuning) for
# image CL. Task 0 is identical across strategies at a given seed, so each run
# reuses the task-0 checkpoint from the matching naive run instead of retraining
# it (worth ~4 GPU-hours per ImageNet run).
#
# Tier 1, the full config grid on the cheap datasets:
#   ./sbatch/image_cl/submit_lora_gft.sh
#
# Tier 2, the selected config carried to ImageNet-100:
#   DATASETS=split_imagenet100 LORA_RANKS=16 GFT_SCHEDULES=sigmoid \
#     ./sbatch/image_cl/submit_lora_gft.sh
#
#   DRY_RUN=1 ./sbatch/image_cl/submit_lora_gft.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH="${REPO_ROOT}/sbatch/image_cl/train_cl.sbatch"

DATASETS=(${DATASETS:-split_fmnist split_cifar10})
SEEDS=(${SEEDS:-42 69 1714})
# Unset falls back to the default grid, empty means "skip this strategy".
LORA_RANKS=(${LORA_RANKS-8 16 32})
GFT_SCHEDULES=(${GFT_SCHEDULES-sigmoid linear})
# Reuse task 0 from this run stem unless REUSE_TASK0=0.
REUSE_TASK0="${REUSE_TASK0:-1}"
TASK0_STEM="${TASK0_STEM:-naive}"
# train_cl.sbatch uses --qos=YOUR_QOS. Set QOS to a queue that will not
# preempt a long ImageNet run.
QOS="${QOS:-YOUR_QOS}"
TIME="${TIME:-}"
DRY_RUN="${DRY_RUN:-0}"
# Skip runs that already finished or are already queued, so a partial campaign
# can be topped up without duplicating in-flight jobs.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

queued_names=""
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  queued_names="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
fi

should_skip() {
  local run_name="$1" job="$2"
  [[ "${SKIP_EXISTING}" != "1" ]] && return 1
  if [[ -f "${REPO_ROOT}/results_dir/${run_name}/metrics.json" ]]; then
    echo "  skip (complete): ${run_name}"
    return 0
  fi
  if grep -qxF "${job}" <<<"${queued_names}"; then
    echo "  skip (queued): ${run_name}"
    return 0
  fi
  return 1
}

slug_for() {
  case "$1" in
    split_cifar10) echo "cifar10" ;;
    split_imagenet100) echo "imagenet100" ;;
    *) echo "fmnist" ;;
  esac
}

n=0
missing=0
for dataset in "${DATASETS[@]}"; do
  slug="$(slug_for "${dataset}")"
  for seed in "${SEEDS[@]}"; do
    task0_run="image_cl/${slug}/${TASK0_STEM}_s${seed}"
    task0_arg=""
    if [[ "${REUSE_TASK0}" == "1" ]]; then
      if [[ -f "${REPO_ROOT}/results_dir/${task0_run}/tasks/00/checkpoint.pt" ]]; then
        task0_arg="${task0_run}"
      else
        echo "  WARNING: no task-0 checkpoint at ${task0_run}; this run will train it" >&2
        missing=$((missing + 1))
      fi
    fi

    for rank in "${LORA_RANKS[@]}"; do
      run_name="image_cl/${slug}/lora_r${rank}_s${seed}"
      export_vars="ALL,STRATEGY=lora,SEED=${seed},RUN_NAME=${run_name}"
      export_vars+=",DATASET=${dataset},RESULTS_SLUG=${slug},REPO_ROOT=${REPO_ROOT}"
      export_vars+=",LORA_RANK=${rank},TASK0_FROM=${task0_arg}"
      job="icl-lora${rank}-${slug:0:4}-s${seed}"
      if should_skip "${run_name}" "${job}"; then continue; fi
      sbatch_args=()
      [[ -n "${QOS}" ]] && sbatch_args+=(--qos="${QOS}")
      [[ -n "${TIME}" ]] && sbatch_args+=(--time="${TIME}")
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  DRY_RUN sbatch --job-name=${job} ${sbatch_args[*]-} --export=${export_vars} ${SBATCH}"
      else
        sbatch --job-name="${job}" "${sbatch_args[@]}" --export="${export_vars}" "${SBATCH}"
      fi
      n=$((n + 1))
    done

    for schedule in "${GFT_SCHEDULES[@]}"; do
      run_name="image_cl/${slug}/gft_${schedule}_s${seed}"
      export_vars="ALL,STRATEGY=gft,SEED=${seed},RUN_NAME=${run_name}"
      export_vars+=",DATASET=${dataset},RESULTS_SLUG=${slug},REPO_ROOT=${REPO_ROOT}"
      export_vars+=",GFT_BETA_SCHEDULE=${schedule},TASK0_FROM=${task0_arg}"
      job="icl-gft-${schedule:0:3}-${slug:0:4}-s${seed}"
      if should_skip "${run_name}" "${job}"; then continue; fi
      sbatch_args=()
      [[ -n "${QOS}" ]] && sbatch_args+=(--qos="${QOS}")
      [[ -n "${TIME}" ]] && sbatch_args+=(--time="${TIME}")
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  DRY_RUN sbatch --job-name=${job} ${sbatch_args[*]-} --export=${export_vars} ${SBATCH}"
      else
        sbatch --job-name="${job}" "${sbatch_args[@]}" --export="${export_vars}" "${SBATCH}"
      fi
      n=$((n + 1))
    done
  done
done

echo "Submitted ${n} jobs across ${DATASETS[*]} (DRY_RUN=${DRY_RUN})."
if [[ "${missing}" != "0" ]]; then
  echo "WARNING: ${missing} seed/dataset pairs had no reusable task 0." >&2
fi
