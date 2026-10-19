#!/bin/bash
#SBATCH --job-name=imagenet-convert
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/imagenet_convert_%j.log

# Convert HF ILSVRC/imagenet-1k parquet → torchvision ImageNet layout, then
# symlink CMC ImageNet-100 class folders.
#
#   sbatch sbatch/image_cl/convert_imagenet.sbatch
#   # ImageNet-100 only (faster):
#   sbatch --export=ALL,MODE=imagenet100 sbatch/image_cl/convert_imagenet.sbatch
#   # Full 1k only (skip symlink):
#   sbatch --export=ALL,MODE=full sbatch/image_cl/convert_imagenet.sbatch
#
# MODE=full_then_symlink (default) | full | imagenet100 | symlink_only
# Idempotent: skips JPEGs that already exist.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

MODE="${MODE:-full_then_symlink}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
HF_RAW="${HF_RAW:-${REPO_ROOT}/data_dir/imagenet/hf_raw}"
IMAGENET_ROOT="${IMAGENET_ROOT:-${REPO_ROOT}/data_dir/imagenet}"
IMAGENET100_ROOT="${IMAGENET100_ROOT:-${REPO_ROOT}/data_dir/imagenet100}"
SYNSETS_FILE="${SYNSETS_FILE:-${REPO_ROOT}/src/trajectory_replay/data/imagenet100_cmc.txt}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} mode=${MODE} workers=${WORKERS}"
echo "hf_raw=${HF_RAW}"
echo "imagenet_root=${IMAGENET_ROOT}"
echo "imagenet100_root=${IMAGENET100_ROOT}"
date -Is

run_convert() {
  local out_root="$1"
  shift
  uv run python scripts/convert_hf_imagenet_to_torchvision.py \
    --hf-raw "${HF_RAW}" \
    --out-root "${out_root}" \
    --workers "${WORKERS}" \
    --splits train val \
    "$@"
}

case "${MODE}" in
  full|full_then_symlink)
    echo "=== converting full ImageNet-1k → ${IMAGENET_ROOT} ==="
    run_convert "${IMAGENET_ROOT}"
    ;;
  imagenet100)
    echo "=== converting CMC ImageNet-100 → ${IMAGENET100_ROOT} ==="
    run_convert "${IMAGENET100_ROOT}" --synsets-file "${SYNSETS_FILE}"
    ;;
  symlink_only)
    echo "=== symlink-only mode (skip convert) ==="
    ;;
  *)
    echo "unknown MODE=${MODE}" >&2
    exit 1
    ;;
esac

if [[ "${MODE}" == "full_then_symlink" || "${MODE}" == "symlink_only" ]]; then
  echo "=== building ImageNet-100 symlinks → ${IMAGENET100_ROOT} ==="
  # Replace any prior real-file tree (e.g. partial login-node / imagenet100 job) with symlinks.
  if [[ -e "${IMAGENET100_ROOT}" ]]; then
    echo "removing existing ${IMAGENET100_ROOT} before symlink pass"
    rm -rf "${IMAGENET100_ROOT}"
  fi
  uv run python scripts/symlink_imagenet100.py \
    --imagenet-root "${IMAGENET_ROOT}" \
    --out-root "${IMAGENET100_ROOT}" \
    --synsets-file "${SYNSETS_FILE}" \
    --force
fi

echo "=== done ==="
date -Is
# Quick sanity counts
for root in "${IMAGENET_ROOT}" "${IMAGENET100_ROOT}"; do
  if [[ -d "${root}/train" ]]; then
    n_cls=$(find "${root}/train" -mindepth 1 -maxdepth 1 -type d -o -type l | wc -l)
    n_img=$(find -L "${root}/train" -type f -name '*.JPEG' | wc -l)
    echo "${root}: train_classes=${n_cls} train_jpegs=${n_img}"
  fi
done
