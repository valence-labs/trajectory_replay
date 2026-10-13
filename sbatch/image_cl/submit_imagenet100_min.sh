#!/usr/bin/env bash
# Submit Split-ImageNet-100 latent FM image_cl grid (class-conditional protocol).
#
# Locked recipe (from e3 class-cond large pilots; 100k beats 200k on FID):
#   class-cond + CFG 1.5, UNet large, lr=1e-4, grad clip=1.0,
#   ~100k Adam steps/task (EPOCHS=667 @ batch 256), n_experiences=3 (34/33/33),
#   Euler step 0.02 (~50 NFEs), cumulative fixed-step schedule,
#   FID test=10000, per-class FID off, QOS=hooke-predict
#
# Run names match CIFAR/FMNIST style (recipe knobs are defaults, not suffixes):
#   naive_s42, cumulative_s42, generative_g3_s42, ...
# Deviations get tags: _uncond, _medium, _e5, _eq, _mf, _g3, ...
#
# Examples (from repo root):
#   # Dry-run finetune + joint upper bound (3 seeds):
#   DRY_RUN=1 STRATEGIES="naive cumulative" ./sbatch/image_cl/submit_imagenet100_min.sh
#
#   # Launch naive + cumulative:
#   STRATEGIES="naive cumulative" ./sbatch/image_cl/submit_imagenet100_min.sh
#
#   # λ-grid gene/traj (ODE replay @30 steps, eval @50 like naive/cumulative):
#   STRATEGIES="generative trajectory" LAMBDA_MODE=grid RUN_SUFFIX=g3 \
#     REPLAY_ODE_STEP_SIZE=0.03333333333333333 TIME=96:00:00 \
#     ./sbatch/image_cl/submit_imagenet100_min.sh
#
#   # MeanFlow replay on gene/traj (after ODE baseline):
#   STRATEGIES="generative trajectory" REPLAY_BACKENDS="mean_flow" \
#     LAMBDA_MODE=grid RUN_SUFFIX=g3_mf TIME=96:00:00 \
#     ./sbatch/image_cl/submit_imagenet100_min.sh
#
#   # Fixed equal-weight λ (replay 30 / eval 50):
#   STRATEGIES="generative trajectory" LAMBDA_MODE=equal RUN_SUFFIX=eq \
#     REPLAY_ODE_STEP_SIZE=0.03333333333333333 TIME=72:00:00 \
#     ./sbatch/image_cl/submit_imagenet100_min.sh
#
#   # Masip 5×20 ablation (scale epochs for ~100k steps on a 20-class task):
#   N_EXPERIENCES=5 EPOCHS=1000 TIME=36:00:00 \
#     STRATEGIES="naive cumulative" ./sbatch/image_cl/submit_imagenet100_min.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET=split_imagenet100
export RESULTS_SLUG="${RESULTS_SLUG:-imagenet100}"

# --- locked quality recipe ---
export CLASS_CONDITIONAL="${CLASS_CONDITIONAL:-1}"
export UNET_SIZE="${UNET_SIZE:-large}"
export LR="${LR:-1e-4}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export ODE_STEP_SIZE="${ODE_STEP_SIZE:-0.02}"
export N_EXPERIENCES="${N_EXPERIENCES:-3}"
# ~100k Adam steps/task on a ~34-class task (150 steps/epoch @ bs 256).
export EPOCHS="${EPOCHS:-667}"
# Keep that step budget fixed across cumulative tasks (shrink epochs as |D| grows).
export CUMULATIVE_SCHEDULE="${CUMULATIVE_SCHEDULE:-steps}"
export LABEL_DROPOUT="${LABEL_DROPOUT:-0.1}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
export FID_TEST_SAMPLES="${FID_TEST_SAMPLES:-10000}"
export FID_VAL_SAMPLES="${FID_VAL_SAMPLES:-2048}"
# Stream FID only for the table; per-class FID off.
export PER_CLASS_FID_SAMPLES="${PER_CLASS_FID_SAMPLES:-0}"
export NO_KLD="${NO_KLD:-1}"
export DISTILL_EPOCHS="${DISTILL_EPOCHS:-150}"

export STRATEGIES="${STRATEGIES:-naive cumulative}"
export SEEDS="${SEEDS:-42 69 1714}"
export REPLAY_BACKENDS="${REPLAY_BACKENDS:-ode}"
export LAMBDA_MODE="${LAMBDA_MODE:-grid}"
export LAMBDA_GRID="${LAMBDA_GRID:-0.25 0.5 0.75}"
export QOS="${QOS:-hooke-predict}"
# Empty by default (naive/cumulative). Set RUN_SUFFIX=g3 for λ-grid gene/traj.
export RUN_SUFFIX="${RUN_SUFFIX:-}"

# Walltime: ~3×4h train + FID under fixed ~100k steps/task (naive and cumulative).
export TIME="${TIME:-24:00:00}"

exec "${REPO_ROOT}/sbatch/image_cl/submit_fmnist_min.sh"
