#!/bin/bash
#SBATCH --job-name=icl-render-grids
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/image_cl_render_grids_%j.log
# CPU-only: rendering saved *_gen.pt tensors to PNG (already VAE-decoded RGB).

# Render image_cl sample tensors to PNG grids.
#
#   sbatch --export=ALL,ROOT=results_dir/image_cl/imagenet100 \
#     sbatch/image_cl/render_sample_grids.sbatch
#   OVERWRITE=1 sbatch --export=ALL,ROOT=results_dir/image_cl/cifar10 \
#     sbatch/image_cl/render_sample_grids.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

: "${ROOT:=results_dir/image_cl/imagenet100}"
: "${NROW:=8}"
: "${OVERWRITE:=0}"
export REPO_ROOT ROOT NROW OVERWRITE

unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

echo "Job ${SLURM_JOB_ID:-local} render_sample_grids at $(date -Is)"
echo "  ROOT=${ROOT} NROW=${NROW} OVERWRITE=${OVERWRITE}"

uv sync

render_args=(experiments/image_cl/render_sample_grids.py --root "${ROOT}" --nrow "${NROW}")
if [[ "${OVERWRITE}" == "1" ]]; then
  render_args+=(--overwrite)
fi
uv run python "${render_args[@]}"

# Also stage a flat review folder with final-task FM + Masip diffusion PNGs.
REVIEW="${ROOT}/_sample_review"
mkdir -p "${REVIEW}"
export REVIEW
uv run python - <<'PY'
import os
import shutil
from pathlib import Path

root = Path(os.environ["ROOT"])
review = Path(os.environ["REVIEW"])
review.mkdir(parents=True, exist_ok=True)

for png in sorted(root.glob("*/samples/task04_gen.png")):
    if "smoke" in str(png):
        continue
    run = png.parents[1].name
    dest = review / f"fm_{run}_task04.png"
    shutil.copy2(png, dest)
    print(f"staged {dest}")

for run in ("naive_s42", "cumulative_s42"):
    src = root / run / "samples" / "task00_gen.png"
    if src.is_file():
        dest = review / f"fm_{run}_task00.png"
        shutil.copy2(src, dest)
        print(f"staged {dest}")

masip = Path(os.environ["REPO_ROOT"]) / "results_dir/baselines/masip/split_imagenet100"
for strat, tag in (
    ("gr_diffusion_naive_steps_20_lambd_0.75", "diff_naive"),
    ("gr_diffusion_cumulative_steps_20_lambd_0.75", "diff_cumulative"),
):
    for seed in ("42", "69", "1714"):
        src = masip / strat / seed / "samples" / "0004.png"
        if src.is_file():
            dest = review / f"{tag}_s{seed}_task04.png"
            shutil.copy2(src, dest)
            print(f"staged {dest}")

print(f"REVIEW={review.resolve()}")
PY

echo "Done at $(date -Is)"
echo "Review folder: ${REVIEW}"
