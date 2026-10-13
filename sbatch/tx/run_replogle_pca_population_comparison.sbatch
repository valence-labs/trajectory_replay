#!/bin/bash
#SBATCH --job-name=replogle-pca-pop-compare
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=sbatch/logs/replogle_pca_pop_compare_%j.log

# Old-only vs old+new-train PCA fitting population comparison on Replogle,
# scored by gene-space reconstruction quality on held-out HepG2 test cells.
#
#   sbatch sbatch/tx/run_replogle_pca_population_comparison.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p sbatch/logs

PYTHON_BIN="${SIM_ROOT}/.venv/bin/python3"

"${PYTHON_BIN}" sbatch/tx/replogle_pca_population_comparison.py
