#!/bin/bash
#SBATCH --job-name=knn-test
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=sbatch/logs/knn_test_%j.log

# KNN classifier sanity check on real MNIST pixels (CPU, no model calls).
#
#   sbatch sbatch/mnist/test_knn_classifier.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${K:=5}"
: "${TRAIN_MAX:=20000}"
: "${TEST_MAX:=5000}"

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "  K=${K}  TRAIN_MAX=${TRAIN_MAX}  TEST_MAX=${TEST_MAX}"

uv sync
uv run python experiments/mnist/test_knn_classifier.py \
  --k "${K}" \
  --train-max "${TRAIN_MAX}" \
  --test-max "${TEST_MAX}"

echo "Finished at $(date -Is)"
