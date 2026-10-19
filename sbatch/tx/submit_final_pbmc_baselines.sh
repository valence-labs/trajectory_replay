#!/usr/bin/env bash
# Final PBMC PCA-100 non-replay matrix: 13 finetuning configs x 3 seeds.
# Training and dependent CellEval jobs run on normal QoS.
#
# DRY_RUN=1 ./sbatch/tx/submit_final_pbmc_baselines.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results/pbmc

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(${SEEDS:-42 69 1714})
DECODER_DIR="${DECODER_DIR:-${REPO_ROOT}/results_dir/tx_results/pbmc/decoders_pca100_affine}"

submit() {
  extra_opts=(--qos=YOUR_QOS ${SBATCH_OPTS:-})
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "sbatch ${extra_opts[*]} $*" >&2
    echo "dry-run"
  else
    sbatch --parsable "${extra_opts[@]}" "$@"
  fi
}

env_job="${ENV_JOB_ID:-}"
train_dep=()
if [[ "${PREPARE_ENV:-1}" == "1" ]]; then
  env_job="$(submit sbatch/tx/prepare_cl_environment.sbatch)"
fi
if [[ -n "${env_job}" && "${DRY_RUN}" != "1" ]]; then
  train_dep=(--dependency="afterok:${env_job}")
fi

n_train=0
n_celleval=0
names=()

submit_one() {
  local strategy="$1" scope="$2" pert="$3" ctx="$4"
  local lora_rank="$5" lora_alpha="$6" gft_sched="$7" tsf="$8"
  local seed run_name out_dir train_export train_job
  local -a ce_dep=()

  for seed in "${SEEDS[@]}"; do
    run_name="${strategy}_pca100"
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
    run_name+="_p${pert}c${ctx}_st300000_ft150000_final_res2048_d6_s${seed}"
    out_dir="results_dir/tx_results/pbmc/${run_name}"
    if [[ "${SKIP_COMPLETED:-0}" == "1" \
      && -f "${out_dir}/final_metrics.json" \
      && -f "${out_dir}/cell_eval_minimal_stream_heldout.json" ]]; then
      echo "Skipping completed baseline: ${run_name}" >&2
      continue
    fi
    names+=("${run_name}")

    train_export="ALL,DATASET=pbmc,STRATEGY=${strategy},REPRESENTATION=pca,PCA_DIM=100,REPLAY_BACKEND=ode,MEAN_FLOW_STEPS=1,LAMBDA_MODE=none,FINETUNE_SCOPE=${scope},FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS=${pert},FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS=${ctx},LORA_RANK=${lora_rank},LORA_ALPHA=${lora_alpha},LORA_TRAIN_OUTPUT_LAYER=0,TWO_STAGE_1_EPOCHS=0,TWO_STAGE_1_EPOCH_FRACTION=${tsf},GFT_BETA_SCHEDULE=${gft_sched},GFT_BETA_START=1.0,GFT_BETA_END=0.0,GFT_SIGMOID_MIDPOINT=0.5,GFT_SIGMOID_STEEPNESS=10.0,SEED=${seed},RUN_NAME=${run_name},OUT_DIR=${out_dir},TARGET_STEPS=300000,LATER_TARGET_STEPS=150000,LR=1e-4,LATER_LR=0,HIDDEN_DIM=2048,DEPTH=6,BACKBONE_TYPE=residual_mlp,NO_RESUME=1,RUN_CELLEVAL=0,RUN_HELDOUT_CURVE=1,DECODER_DIR=${DECODER_DIR},REQUIRE_EXISTING_DECODERS=1,PRED_CONTROL=decoded,EVAL_MAX_CELLS_PER_STRATUM=0,EVAL_MAX_CONTROL_CELLS=0,EVAL_GEN_PER_STRATUM=500,LAMBDA_GEN_PER_STRATUM=200,EVAL_FINAL_TRACKS=stream+heldout"
    train_job="$(submit "${train_dep[@]}" --job-name="tx-pb-${run_name}" \
      --export="${train_export}" sbatch/tx/train_cl.sbatch)"
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
      --job-name="tx-pb-ce-${run_name}" \
      --export="ALL,OUT_DIR=${out_dir},HELDOUT_CURVE=0,TRACKS=stream+heldout" \
      sbatch/tx/eval_cl_celleval.sbatch >/dev/null
    n_celleval=$((n_celleval + 1))
  done
}

# Naive freeze grid (vf=0 -> embedding scope, vf=1 -> full scope).
submit_one standard embedding 0 0 8 16 sigmoid ""
submit_one standard embedding 1 1 8 16 sigmoid ""
submit_one standard full 0 0 8 16 sigmoid ""
submit_one standard full 0 1 8 16 sigmoid ""
submit_one standard full 1 0 8 16 sigmoid ""
submit_one standard full 1 1 8 16 sigmoid ""

# LoRA, GFT, and two-stage baselines.
submit_one standard lora 0 0 4 8 sigmoid ""
submit_one standard lora 0 0 8 16 sigmoid ""
submit_one standard lora 0 0 16 32 sigmoid ""
submit_one gft full 0 0 8 16 sigmoid ""
submit_one gft full 0 0 8 16 linear ""
submit_one standard full 0 0 8 16 sigmoid 0.1
submit_one standard full 0 0 8 16 sigmoid 0.5

echo "Submitted ${n_train} baseline training jobs and ${n_celleval} CellEval jobs (DRY_RUN=${DRY_RUN}, QOS=normal)."
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${names[@]}" | sort
  echo "unique names: $(printf '%s\n' "${names[@]}" | sort -u | wc -l)"
fi
