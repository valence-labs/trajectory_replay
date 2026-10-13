#!/usr/bin/env bash
# Submit the complete sequential Tx CL campaign and post-hoc evaluations.
#
# DRY_RUN=1 ./sbatch/tx/submit_cl_campaign.sh
# DATASETS="replogle" SEEDS="42" ./sbatch/tx/submit_cl_campaign.sh
# REPRESENTATION=pca DATASETS="replogle" SEEDS="42" ./sbatch/tx/submit_cl_campaign.sh
# REPRESENTATION=pca PREPARE_DECODERS=0 DATASETS="replogle" SEEDS="69 1714" ./sbatch/tx/submit_cl_campaign.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs results_dir/tx_results

DATASETS=(${DATASETS:-replogle pbmc})
STRATEGIES=(${STRATEGIES:-standard gft real generative trajectory})
FINETUNE_SCOPES=(${FINETUNE_SCOPES:-full})
FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS="${FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS:-0}"
FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS="${FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS:-0}"
REPLAY_BACKENDS=(${REPLAY_BACKENDS:-ode mean_flow})
MEAN_FLOW_STEPS_GRID=(${MEAN_FLOW_STEPS_GRID:-1 4})
LAMBDA_MODES=(${LAMBDA_MODES:-grid equal})
SEEDS=(${SEEDS:-42 69 1714})
REPRESENTATION="${REPRESENTATION:-se}"
# Canonical PCA width; must match DEFAULT_PCA_DIM in tx_preprocess.py.
# Unlabeled data_dir/replogle/pca_* is 256; other widths are pca{d}_*.
PCA_DIM="${PCA_DIM:-256}"
PREPARE_DECODERS="${PREPARE_DECODERS:-1}"
# After protocol changes, never continue an old checkpoint in the same run dir.
NO_RESUME="${NO_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
TWO_STAGE_1_EPOCHS="${TWO_STAGE_1_EPOCHS:-0}"
TWO_STAGE_1_EPOCH_FRACTION="${TWO_STAGE_1_EPOCH_FRACTION:-}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TRAIN_OUTPUT_LAYER="${LORA_TRAIN_OUTPUT_LAYER:-0}"
GFT_BETA_SCHEDULE="${GFT_BETA_SCHEDULE:-sigmoid}"
# Pearson heldout curve stays in the train job. Cell-Eval is a separate
# afterok GPU job so a Cell-Eval preemption cannot wipe the trainer.
RUN_CELLEVAL="${RUN_CELLEVAL:-0}"
SUBMIT_CELLEVAL="${SUBMIT_CELLEVAL:-1}"
RUN_HELDOUT_CURVE="${RUN_HELDOUT_CURVE:-1}"
PREPARE_ENV="${PREPARE_ENV:-${SUBMIT_CELLEVAL}}"
TARGET_STEPS="${TARGET_STEPS:-100000}"
# Tasks after T0. 0 means inherit TARGET_STEPS. Replogle T1 is 50k so HepG2
# is not a second full pretrain (~3226 epochs at 100k).
LATER_TARGET_STEPS="${LATER_TARGET_STEPS:-0}"
LR="${LR:-1e-4}"
LATER_LR="${LATER_LR:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
DEPTH="${DEPTH:-4}"
BACKBONE_TYPE="${BACKBONE_TYPE:-mlp}"
SBATCH_TIME="${SBATCH_TIME:-}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.05}"
TRAJECTORY_BATCH_SAMPLING="${TRAJECTORY_BATCH_SAMPLING:-equal}"
TRAJECTORY_REPLAY_BATCH_MULTIPLIER="${TRAJECTORY_REPLAY_BATCH_MULTIPLIER:-1.0}"
TRAJECTORY_REPLAY_LOSS_MULTIPLIER="${TRAJECTORY_REPLAY_LOSS_MULTIPLIER:-1.0}"
TRAJECTORY_TIME_SAMPLING="${TRAJECTORY_TIME_SAMPLING:-batch}"

if [[ ! "${TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_STEPS must be a positive integer; got '${TARGET_STEPS}'" >&2
  exit 2
fi
if [[ -n "${LATER_TARGET_STEPS}" && "${LATER_TARGET_STEPS}" != "0" && ! "${LATER_TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LATER_TARGET_STEPS must be 0 or a positive integer; got '${LATER_TARGET_STEPS}'" >&2
  exit 2
fi

submit() {
  extra_opts=(${SBATCH_OPTS:-})
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "sbatch ${extra_opts[*]} $*" >&2
    echo "dry-run"
  else
    sbatch --parsable "${extra_opts[@]}" "$@"
  fi
}

submit_env() {
  extra_opts=(${ENV_SBATCH_OPTS:-${SBATCH_OPTS:-}})
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "sbatch ${extra_opts[*]} $*" >&2
    echo "dry-run"
  else
    sbatch --parsable "${extra_opts[@]}" "$@"
  fi
}

n_train=0
n_celleval=0
echo "REPRESENTATION=${REPRESENTATION} PCA_DIM=${PCA_DIM} TARGET_STEPS=${TARGET_STEPS} LATER_TARGET_STEPS=${LATER_TARGET_STEPS} PREPARE_ENV=${PREPARE_ENV} PREPARE_DECODERS=${PREPARE_DECODERS} RUN_CELLEVAL=${RUN_CELLEVAL} SUBMIT_CELLEVAL=${SUBMIT_CELLEVAL} RUN_HELDOUT_CURVE=${RUN_HELDOUT_CURVE}"
train_dependency=()
env_job="${ENV_JOB_ID:-}"
if [[ "${PREPARE_ENV}" == "1" ]]; then
  env_job="$(submit_env sbatch/tx/prepare_cl_environment.sbatch)"
fi
if [[ -n "${env_job}" && "${DRY_RUN}" != "1" ]]; then
  train_dependency=(--dependency="afterok:${env_job}")
fi

for dataset in "${DATASETS[@]}"; do
  dataset_dependency=("${train_dependency[@]}")
  if [[ "${PREPARE_DECODERS}" == "1" ]]; then
    decoder_export="ALL,DATASET=${dataset},REPRESENTATION=${REPRESENTATION}"
    if [[ "${REPRESENTATION}" == "pca" ]]; then
      decoder_export+=",PCA_DIM=${PCA_DIM}"
    fi
    decoder_job="$(submit "${train_dependency[@]}" --export="${decoder_export}" \
      sbatch/tx/prepare_cl_decoders.sbatch)"
    if [[ "${DRY_RUN}" != "1" ]]; then
      dataset_dependency=(--dependency="afterok:${decoder_job}")
    fi
  fi

  for strategy in "${STRATEGIES[@]}"; do
    if [[ "${strategy}" == "generative" || "${strategy}" == "trajectory" ]]; then
      strategy_backends=("${REPLAY_BACKENDS[@]}")
    else
      strategy_backends=(ode)
    fi
    if [[ "${strategy}" == "real" || "${strategy}" == "generative" || "${strategy}" == "trajectory" ]]; then
      strategy_modes=("${LAMBDA_MODES[@]}")
    else
      strategy_modes=(none)
    fi
    for backend in "${strategy_backends[@]}"; do
      if [[ "${backend}" == "mean_flow" ]]; then
        steps_grid=("${MEAN_FLOW_STEPS_GRID[@]}")
      else
        steps_grid=(1)
      fi
      for steps in "${steps_grid[@]}"; do
        backend_tag="ode"
        if [[ "${backend}" == "mean_flow" ]]; then
          backend_tag="mf_n${steps}"
        fi
        for mode in "${strategy_modes[@]}"; do
          for scope in "${FINETUNE_SCOPES[@]}"; do
            for seed in "${SEEDS[@]}"; do
              run_name="${strategy}"
              if [[ "${REPRESENTATION}" == "genes" ]]; then
                run_name+="_gene"
              elif [[ "${REPRESENTATION}" == "scimilarity" ]]; then
                run_name+="_sci"
              elif [[ "${REPRESENTATION}" == "pca" ]]; then
                run_name+="_pca${PCA_DIM}"
              fi
              if [[ "${strategy}" == "generative" || "${strategy}" == "trajectory" ]]; then
                run_name+="_${backend_tag}"
              fi
              if [[ "${mode}" != "none" ]]; then
                run_name+="_${mode}"
              fi
              if [[ "${scope}" != "full" ]]; then
                run_name+="_${scope}"
              fi
              if [[ "${scope}" == "lora" ]]; then
                run_name+="_r${LORA_RANK}"
              fi
              if [[ "${strategy}" == "gft" ]]; then
                run_name+="_${GFT_BETA_SCHEDULE:-sigmoid}"
              fi
              if [[ "${TWO_STAGE_1_EPOCHS}" != "0" ]]; then
                run_name+="_ts${TWO_STAGE_1_EPOCHS}"
              elif [[ -n "${TWO_STAGE_1_EPOCH_FRACTION}" ]]; then
                run_name+="_tsf${TWO_STAGE_1_EPOCH_FRACTION}"
              fi
              if [[ "${strategy}" != "real" && "${strategy}" != "generative" && "${strategy}" != "trajectory" ]]; then
                run_name+="_p${FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS}c${FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS}"
              fi
              run_name+="_st${TARGET_STEPS}"
              if [[ "${LATER_TARGET_STEPS}" =~ ^[1-9][0-9]*$ && "${LATER_TARGET_STEPS}" != "${TARGET_STEPS}" ]]; then
                run_name+="_ft${LATER_TARGET_STEPS}"
              fi
              if [[ -n "${RUN_NAME_SUFFIX}" ]]; then
                run_name+="_${RUN_NAME_SUFFIX}"
              fi
              run_name+="_s${seed}"
              out_dir="results_dir/tx_results/${dataset}/${run_name}"
              train_time=()
              if [[ -n "${SBATCH_TIME}" ]]; then
                train_time=(--time="${SBATCH_TIME}")
              fi
              train_export="ALL,DATASET=${dataset},STRATEGY=${strategy},REPRESENTATION=${REPRESENTATION},REPLAY_BACKEND=${backend},MEAN_FLOW_STEPS=${steps},LAMBDA_MODE=${mode},FINETUNE_SCOPE=${scope},FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS=${FINETUNE_TRAIN_OLD_PERT_EMBEDDINGS},FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS=${FINETUNE_TRAIN_OLD_CONTEXT_EMBEDDINGS},LORA_RANK=${LORA_RANK},LORA_ALPHA=${LORA_ALPHA},LORA_TRAIN_OUTPUT_LAYER=${LORA_TRAIN_OUTPUT_LAYER},TWO_STAGE_1_EPOCHS=${TWO_STAGE_1_EPOCHS},TWO_STAGE_1_EPOCH_FRACTION=${TWO_STAGE_1_EPOCH_FRACTION},GFT_BETA_SCHEDULE=${GFT_BETA_SCHEDULE},SEED=${seed},RUN_NAME=${run_name},OUT_DIR=${out_dir},TARGET_STEPS=${TARGET_STEPS},LATER_TARGET_STEPS=${LATER_TARGET_STEPS},LR=${LR},LATER_LR=${LATER_LR},HIDDEN_DIM=${HIDDEN_DIM},DEPTH=${DEPTH},BACKBONE_TYPE=${BACKBONE_TYPE},TRAJECTORY_BATCH_SAMPLING=${TRAJECTORY_BATCH_SAMPLING},TRAJECTORY_REPLAY_BATCH_MULTIPLIER=${TRAJECTORY_REPLAY_BATCH_MULTIPLIER},TRAJECTORY_REPLAY_LOSS_MULTIPLIER=${TRAJECTORY_REPLAY_LOSS_MULTIPLIER},TRAJECTORY_TIME_SAMPLING=${TRAJECTORY_TIME_SAMPLING},NO_RESUME=${NO_RESUME},RUN_CELLEVAL=${RUN_CELLEVAL},RUN_HELDOUT_CURVE=${RUN_HELDOUT_CURVE}"
              if [[ "${REPRESENTATION}" == "pca" ]]; then
                train_export+=",PCA_DIM=${PCA_DIM}"
              fi
              if [[ -n "${DECODER_DIR:-}" ]]; then
                train_export+=",DECODER_DIR=${DECODER_DIR}"
              fi
              if [[ "${REQUIRE_EXISTING_DECODERS:-0}" == "1" ]]; then
                train_export+=",REQUIRE_EXISTING_DECODERS=1"
              fi
              train_export+=",PRED_CONTROL=${PRED_CONTROL:-decoded}"
              train_export+=",EVAL_MAX_CELLS_PER_STRATUM=${EVAL_MAX_CELLS_PER_STRATUM:-0}"
              train_export+=",EVAL_MAX_CONTROL_CELLS=${EVAL_MAX_CONTROL_CELLS:-0}"
              train_export+=",EVAL_GEN_PER_STRATUM=${EVAL_GEN_PER_STRATUM:-500}"
              train_export+=",LAMBDA_GEN_PER_STRATUM=${LAMBDA_GEN_PER_STRATUM:-100}"
              train_export+=",EVAL_FINAL_TRACKS=${EVAL_FINAL_TRACKS:-fit+forget+stream+heldout}"
              train_export+=",REPLAY_CONDITION_SAMPLING=${REPLAY_CONDITION_SAMPLING:-proportional}"
              train_export+=",ODE_STEP_SIZE=${ODE_STEP_SIZE}"
              train_job="$(submit "${train_time[@]}" "${dataset_dependency[@]}" \
                --job-name="tx-${dataset:0:3}-${strategy:0:2}-${backend_tag}-${mode:0:1}-${scope:0:2}-${seed}" \
                --export="${train_export}" \
                sbatch/tx/train_cl.sbatch)"
              n_train=$((n_train + 1))
              if [[ "${SUBMIT_CELLEVAL}" == "1" ]]; then
                ce_dep=()
                if [[ "${DRY_RUN}" != "1" ]]; then
                  if [[ -n "${env_job}" ]]; then
                    ce_dep=(--dependency="afterok:${env_job}:${train_job}")
                  else
                    ce_dep=(--dependency="afterok:${train_job}")
                  fi
                fi
                ce_export="ALL,OUT_DIR=${out_dir},HELDOUT_CURVE=${CELLEVAL_HELDOUT_CURVE:-both}"
                if [[ -n "${TRACKS:-}" ]]; then
                  ce_export+=",TRACKS=${TRACKS}"
                fi
                submit "${ce_dep[@]}" \
                  --job-name="tx-cl-ce-${strategy:0:2}-${backend_tag}-${mode:0:1}-${seed}" \
                  --export="${ce_export}" \
                  sbatch/tx/eval_cl_celleval.sbatch >/dev/null
                n_celleval=$((n_celleval + 1))
              fi
            done
          done
        done
      done
    done
  done
done

echo "Submitted ${n_train} training jobs and ${n_celleval} Cell-Eval afterok jobs (DRY_RUN=${DRY_RUN})."
