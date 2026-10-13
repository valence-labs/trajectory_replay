#!/usr/bin/env bash
# Build fit / forget / OOD Cell-Eval tables for all methods (single *_all.tsv manifest only).
#
#   ./sbatch/tx/run_celleval_all_methods_tables.sh
#   BUCKETS="pbmc_split3_state" ./sbatch/tx/run_celleval_all_methods_tables.sh
#   BUCKETS="replogle_split2_hepg2" ./sbatch/tx/run_celleval_all_methods_tables.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SIM_ROOT}"

BUCKETS="${BUCKETS:-tx_results_old/pbmc_split3_state tx_results_old/replogle_split2_hepg2}"

for bucket in ${BUCKETS}; do
  case "${bucket}" in
    pbmc_split3_state|tx_results_old/pbmc_split3_state)
      TX_RESULTS_BUCKET="${bucket}" RUN_LOCAL=1 "${SCRIPT_DIR}/submit_plot_pbmc_split3_state_celleval_analysis.sh"
      ;;
    replogle_split2_hepg2|tx_results_old/replogle_split2_hepg2)
      TX_RESULTS_BUCKET="${bucket}" RUN_LOCAL=1 "${SCRIPT_DIR}/submit_plot_replogle_split2_hepg2_celleval_analysis.sh"
      ;;
    *)
      echo "ERROR: unknown bucket ${bucket}" >&2
      exit 1
      ;;
  esac
done

echo "All-methods Cell-Eval tables written under results_dir/<bucket>/metrics_celleval/"
