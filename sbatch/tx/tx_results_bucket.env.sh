# Source from tx sbatch/submit scripts (after SCRIPT_DIR is set).
# Override per campaign: export TX_RESULTS_BUCKET=ho_d8_indistrib_small
export TX_RESULTS_BUCKET="${TX_RESULTS_BUCKET:-ho_d2_indistrib_small}"
export RESULTS_DIR="${RESULTS_DIR:-./results_dir/${TX_RESULTS_BUCKET}}"
