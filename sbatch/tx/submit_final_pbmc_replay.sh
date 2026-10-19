#!/usr/bin/env bash
# Final PBMC PCA-100 replay matrix: 14 methods x 3 seeds.
# Training and dependent CellEval jobs use YOUR_QOS from the sbatch headers.
#
# DRY_RUN=1 ./sbatch/tx/submit_final_pbmc_replay.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export STRATEGIES="real generative trajectory"
export REPLAY_BACKENDS="ode mean_flow"
export MEAN_FLOW_STEPS_GRID="1 4"
export LAMBDA_MODES="grid equal"
export SEEDS="${SEEDS:-42 69 1714}"

export TARGET_STEPS=300000
export LATER_TARGET_STEPS=150000
export LR=1e-4
export LATER_LR=0
export HIDDEN_DIM=2048
export DEPTH=6
export BACKBONE_TYPE=residual_mlp
export RUN_NAME_SUFFIX=final_res2048_d6

export REPLAY_CONDITION_SAMPLING=proportional
export ODE_STEP_SIZE=0.05
export TRAJECTORY_BATCH_SAMPLING=equal
export TRAJECTORY_REPLAY_BATCH_MULTIPLIER=1.0
export TRAJECTORY_REPLAY_LOSS_MULTIPLIER=1.0
export TRAJECTORY_TIME_SAMPLING=batch

export EVAL_GEN_PER_STRATUM=500
export LAMBDA_GEN_PER_STRATUM=200
export PREPARE_DECODERS=0
export REQUIRE_EXISTING_DECODERS=1
export PREPARE_ENV="${PREPARE_ENV:-1}"
export SUBMIT_CELLEVAL=1
export NO_RESUME=1
export SBATCH_OPTS="--qos=YOUR_QOS"
export ENV_SBATCH_OPTS="--qos=YOUR_QOS"

exec "${REPO_ROOT}/sbatch/tx/submit_cl_pca100_pbmc.sh" "$@"
