#!/usr/bin/env bash
# Final Replogle PCA-50 replay matrix: 14 methods x 3 seeds.
# Training and dependent CellEval jobs run on hooke-predict.
#
# DRY_RUN=1 ./sbatch/tx/submit_final_replogle_replay.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export STRATEGIES="real generative trajectory"
export REPLAY_BACKENDS="ode mean_flow"
export MEAN_FLOW_STEPS_GRID="1 4"
export LAMBDA_MODES="grid equal"
export SEEDS="${SEEDS:-42 69 1714}"

export TARGET_STEPS=100000
export LATER_TARGET_STEPS=50000
export LR=1e-4
export LATER_LR=0
export HIDDEN_DIM=2048
export DEPTH=4
export BACKBONE_TYPE=mlp
export RUN_NAME_SUFFIX=final_mlp2048_d4

export REPLAY_CONDITION_SAMPLING=proportional
export ODE_STEP_SIZE=0.05
export TRAJECTORY_BATCH_SAMPLING=equal
export TRAJECTORY_REPLAY_BATCH_MULTIPLIER=1.0
export TRAJECTORY_REPLAY_LOSS_MULTIPLIER=1.0
export TRAJECTORY_TIME_SAMPLING=batch

export EVAL_GEN_PER_STRATUM=500
export LAMBDA_GEN_PER_STRATUM=200
export EVAL_FINAL_TRACKS=stream+heldout
export TRACKS=stream+heldout
export PREPARE_DECODERS=0
export REQUIRE_EXISTING_DECODERS=1
export PREPARE_ENV="${PREPARE_ENV:-1}"
export SUBMIT_CELLEVAL=1
export CELLEVAL_HELDOUT_CURVE=0
export NO_RESUME=1
export SBATCH_OPTS="--qos=hooke-predict --exclude=hop63"
export ENV_SBATCH_OPTS="--qos=normal"

exec "${REPO_ROOT}/sbatch/tx/submit_cl_pca50.sh" "$@"
