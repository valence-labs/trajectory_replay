#!/usr/bin/env bash
# Submit the full three-seed FM matrix with one shared task-0 checkpoint per seed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-42 69 1714}"
STEPS_PER_TASK="${STEPS_PER_TASK:-25000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_dir/px/cl_final}"

submit_method() {
  local dependency="$1"
  local seed="$2"
  local strategy="$3"
  local backend="$4"
  local lambda_mode="$5"
  local mf_steps="$6"
  local task0_dir="${REPO_ROOT}/${OUTPUT_ROOT}/task0_s${seed}"
  local stem="${strategy}_${backend}"
  if [[ "${backend}" == "mean_flow" ]]; then
    stem="${stem}_n${mf_steps}"
  fi
  stem="${stem}_${lambda_mode}_s${seed}"
  local exports="ALL,STRATEGY=${strategy},SEED=${seed},REPLAY_BACKEND=${backend}"
  exports+=",MEAN_FLOW_N_STEPS=${mf_steps},LAMBDA_MODE=${lambda_mode}"
  exports+=",STEPS_PER_TASK=${STEPS_PER_TASK},BATCH_SIZE=${BATCH_SIZE}"
  exports+=",TASK0_CHECKPOINT=${task0_dir}/checkpoint.ckpt"
  exports+=",OUTPUT_DIR=${OUTPUT_ROOT}/${stem}"
  if [[ "${backend}" == "mean_flow" ]]; then
    exports+=",TASK0_MEAN_FLOW=${task0_dir}/mean_flow.pt"
  fi
  sbatch --dependency="afterok:${dependency}" --export="${exports}" \
    sbatch/px/train_cl.sbatch
}

for seed in ${SEEDS}; do
  task0_dir="${REPO_ROOT}/${OUTPUT_ROOT}/task0_s${seed}"
  task0_job="$(
    sbatch --parsable \
      --export="ALL,STRATEGY=trajectory,SEED=${seed},REPLAY_BACKEND=mean_flow,MAX_TASKS=1,EVAL_FID=0,STEPS_PER_TASK=${STEPS_PER_TASK},BATCH_SIZE=${BATCH_SIZE},EXPORT_TASK0_DIR=${task0_dir},OUTPUT_DIR=${OUTPUT_ROOT}/task0_source_s${seed}" \
      sbatch/px/train_cl.sbatch
  )"
  echo "seed=${seed} task0_job=${task0_job}"

  submit_method "${task0_job}" "${seed}" naive ode equal 1
  submit_method "${task0_job}" "${seed}" cumulative ode equal 1
  submit_method "${task0_job}" "${seed}" real ode equal 1

  for strategy in generative trajectory; do
    for lambda_mode in equal grid; do
      submit_method "${task0_job}" "${seed}" "${strategy}" ode "${lambda_mode}" 1
      for mf_steps in 1 4; do
        submit_method \
          "${task0_job}" "${seed}" "${strategy}" mean_flow "${lambda_mode}" "${mf_steps}"
      done
    done
  done
done
