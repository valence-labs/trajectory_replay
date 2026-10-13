#!/bin/bash
#SBATCH --job-name=sci-norm-diag
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=sbatch/logs/sci_norm_diag_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# align-only vs align+tp10k SCimilarity encoding, on a subsample.
#
#   sbatch sbatch/tx/diag_sci_normalization.sbatch

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs

SCI_PYTHON="${SCI_PYTHON:-${REPO_ROOT}/third_party/scimilarity/.venv/bin/python}"

"${SCI_PYTHON}" experiments/tx/diag_sci_normalization.py "$@"
