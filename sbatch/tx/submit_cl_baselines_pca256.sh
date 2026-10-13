#!/usr/bin/env bash
# 13 PCA-256 baseline configs × 3 seeds (naive / LoRA / GFT / two-stage).
# Locked to the official CL protocol: 100k T0 / 50k T1, shared affine decoder bank.
#
#   DRY_RUN=1 ./sbatch/tx/submit_cl_baselines_pca256.sh
#   ./sbatch/tx/submit_cl_baselines_pca256.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/replogle

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(${SEEDS:-42 69 1714})
TARGET_STEPS="${TARGET_STEPS:-100000}"
LATER_TARGET_STEPS="${LATER_TARGET_STEPS:-50000}"

submit() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "sbatch $*" >&2
    echo "dry-run"
  else
    sbatch --parsable "$@"
  fi
}

env_job=""
if [[ "${DRY_RUN}" != "1" ]]; then
  env_job="$(submit sbatch/tx/prepare_cl_environment.sbatch)"
fi

n_train=0
n_celleval=0
names=()

submit_one() {
  local strategy="$1" scope="$2" pert="$3" ctx="$4"
  local lora_rank="$5" lora_alpha="$6" gft_sched="$7" tsf="$8"
  local seed run_name out_dir train_export train_job ce_dep
  for seed in "${SEEDS[@]}"; do
    run_name="${strategy}_pca256"
    if [[ "${scope}" != "full" ]]; then
      run_name+="_${scope}"
    fi
    if [[ "${scope}" == "lora" ]]; then
      run_name+="_r${lora_rank}"
    fi
    if [[ "${strategy}" == "gft" ]]; then
      run_name+="_${gft_sched}"
    fi
    if [[ -n "${tsf}" ]]; then
      run_name+="_tsf${tsf}"
    fi
    run_name+="_p${pert}c${ctx}_st${TARGET_STEPS}_ft${LATER_TARGET_STEPS}_s${seed}"
    out_dir="results_dir/tx_results/replogle/${run_name}"
    names+=("${run_name}")
    train_export="ALL,DATASET=replogle,STRATEGY=${strategy},REPRESENTATION=pca,PCA_DIM=256,REPLAY_BACKEND=ode,MEAN_FLOW_STEPS=1,LAMBDA_MODE=none,FINETUNE_SCOPE=${scope},FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS=${pert},FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS=${ctx},LORA_RANK=${lora_rank},LORA_ALPHA=${lora_alpha},LORA_TRAIN_OUTPUT_LAYER=0,TWO_STAGE_1_EPOCHS=0,TWO_STAGE_1_EPOCH_FRACTION=${tsf},GFT_BETA_SCHEDULE=${gft_sched},GFT_BETA_START=1.0,GFT_BETA_END=0.0,GFT_SIGMOID_MIDPOINT=0.5,GFT_SIGMOID_STEEPNESS=10.0,SEED=${seed},RUN_NAME=${run_name},OUT_DIR=${out_dir},TARGET_STEPS=${TARGET_STEPS},LATER_TARGET_STEPS=${LATER_TARGET_STEPS},NO_RESUME=1,RUN_CELLEVAL=0,RUN_HELDOUT_CURVE=1"
    train_job="$(submit --job-name="tx-bsl-${strategy:0:3}-${scope:0:2}-${seed}" \
      --export="${train_export}" \
      sbatch/tx/train_cl.sbatch)"
    n_train=$((n_train + 1))
    ce_dep=()
    if [[ "${DRY_RUN}" != "1" ]]; then
      if [[ -n "${env_job}" ]]; then
        ce_dep=(--dependency="afterok:${env_job}:${train_job}")
      else
        ce_dep=(--dependency="afterok:${train_job}")
      fi
    fi
    submit "${ce_dep[@]}" \
      --job-name="tx-bsl-ce-${strategy:0:3}-${seed}" \
      --export="ALL,OUT_DIR=${out_dir},HELDOUT_CURVE=both" \
      sbatch/tx/eval_cl_celleval.sbatch >/dev/null
    n_celleval=$((n_celleval + 1))
  done
}

# naive freeze grid (vf=0 -> embedding, vf=1 -> full)
submit_one standard embedding 0 0 8 16 sigmoid ""
submit_one standard embedding 1 1 8 16 sigmoid ""
submit_one standard full 0 0 8 16 sigmoid ""
submit_one standard full 0 1 8 16 sigmoid ""
submit_one standard full 1 0 8 16 sigmoid ""
submit_one standard full 1 1 8 16 sigmoid ""
# LoRA (base VF frozen, old embeddings frozen)
submit_one standard lora 0 0 4 8 sigmoid ""
submit_one standard lora 0 0 8 16 sigmoid ""
submit_one standard lora 0 0 16 32 sigmoid ""
# GFT (full VF, β 1→0)
submit_one gft full 0 0 8 16 sigmoid ""
submit_one gft full 0 0 8 16 linear ""
# two-stage (new ctx, then full VF)
submit_one standard full 0 0 8 16 sigmoid 0.1
submit_one standard full 0 0 8 16 sigmoid 0.5

echo "Submitted ${n_train} training jobs and ${n_celleval} Cell-Eval afterok jobs (DRY_RUN=${DRY_RUN})."
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${names[@]}" | sort
  echo "unique names: $(printf '%s\n' "${names[@]}" | sort -u | wc -l)"
fi
