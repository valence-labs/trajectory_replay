#!/usr/bin/env bash
#SBATCH --job-name=px-sampler-compare
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/px_sampler_compare_%j.log

set -euo pipefail
: "${CHECKPOINT:?Set CHECKPOINT to a TrainState checkpoint}"
: "${OUTPUT_DIR:=results_dir/px/images/adm_batch64_step25000_sampler_comparison}"
: "${BATCH_SIZE:=8}"

cd "${SLURM_SUBMIT_DIR}"
srun uv run python experiments/px/compare_checkpoint_samplers.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --euler-steps 20 \
  --legacy-cell-conditioning
