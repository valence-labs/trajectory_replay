#!/usr/bin/env bash
# Submit matched three-seed Masip diffusion baselines with shared task 0.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-42 69 1714}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_dir/px/masip_final}"
DRY_RUN="${DRY_RUN:-0}"

for seed in ${SEEDS}; do
  task0_dir="${REPO_ROOT}/${OUTPUT_ROOT}/task0_s${seed}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    task0_job="DRY-TASK0-${seed}"
    echo "DRY_RUN sbatch --parsable" \
      --export="ALL,STRATEGY=finetune,SEED=${seed},BATCH_SIZE=64,MAX_TASKS=1,EVAL_FID=0,EXPORT_TASK0_DIR=${task0_dir},OUTPUT_DIR=${OUTPUT_ROOT}/task0_source_s${seed}" \
      sbatch/px/train_masip_cl.sbatch
  else
    task0_job="$(
      sbatch --parsable \
        --export="ALL,STRATEGY=finetune,SEED=${seed},BATCH_SIZE=64,MAX_TASKS=1,EVAL_FID=0,EXPORT_TASK0_DIR=${task0_dir},OUTPUT_DIR=${OUTPUT_ROOT}/task0_source_s${seed}" \
        sbatch/px/train_masip_cl.sbatch
    )"
  fi
  echo "seed=${seed} diffusion_task0_job=${task0_job}"
  for strategy in finetune cumulative generative_replay generative_distillation; do
    batch_size=64
    if [[ "${strategy}" == "generative_replay" || "${strategy}" == "generative_distillation" ]]; then
      batch_size=32
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY_RUN sbatch --dependency=afterok:${task0_job}" \
        --export="ALL,STRATEGY=${strategy},SEED=${seed},BATCH_SIZE=${batch_size},TASK0_CHECKPOINT=${task0_dir}/checkpoint.ckpt,OUTPUT_DIR=${OUTPUT_ROOT}/${strategy}_s${seed}" \
        sbatch/px/train_masip_cl.sbatch
    else
      sbatch \
        --dependency="afterok:${task0_job}" \
        --export="ALL,STRATEGY=${strategy},SEED=${seed},BATCH_SIZE=${batch_size},TASK0_CHECKPOINT=${task0_dir}/checkpoint.ckpt,OUTPUT_DIR=${OUTPUT_ROOT}/${strategy}_s${seed}" \
        sbatch/px/train_masip_cl.sbatch
    fi
  done
done
