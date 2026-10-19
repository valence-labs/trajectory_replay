#!/bin/bash
#SBATCH --job-name=encode-se
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/encode_se_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Encode h5ad(s) with Arc SE-600M via third_party/arc-state (isolated UV env).
#
# From repo root:
#   sbatch sbatch/third_party/encode_se.sbatch
#
# Default input is data_dir/replogle.h5ad inside this repository.
# (same Arc Replogle-Nadig-Preprint file; no HF data re-download).
#
# Env overrides:
#   MODEL_DIR       — SE-600M folder (default: data/arc_state/SE-600M)
#   MODEL_REPO      — HF model repo id (default: arcinstitute/SE-600M)
#   CHECKPOINT      — ckpt path or filename under MODEL_DIR (default: se600m_epoch16.ckpt)
#   INPUT_H5AD      — input AnnData (default: data_dir/replogle.h5ad)
#   OUTPUT_H5AD     — output path (default: results_dir/arc_state/se/<stem>_se.h5ad)
#   OUT_DIR         — used when OUTPUT_H5AD unset (default: results_dir/arc_state/se)
#   EMBED_KEY       — obsm key (default: X_state)
#   BATCH_SIZE      — forward batch size (default: 256; 512 OOMs on 80GB H100 for SE-600M)
#   SKIP_DOWNLOAD   — set to 1 to skip SE-600M HF download
#   DOWNLOAD_DATA   — set to 1 to also pull DATA_REPO into DATA_DIR (off by default)
#   DATA_REPO       — HF dataset (default: arcinstitute/Replogle-Nadig-Preprint)
#   DATA_DIR        — local dir for optional DATA_REPO download
#   DRY_RUN         — set to 1 to print resolved paths and exit
#
# Large datasets (e.g. PBMC ~10M cells): prefer .npy output (embeddings only) and
# raise --mem (encode loads full AnnData + embeddings into RAM):
#   SKIP_DOWNLOAD=1 BATCH_SIZE=256 \\
#     INPUT_H5AD=data_dir/pbmc_parse_with_splits.h5ad \\
#     OUTPUT_H5AD=results_dir/arc_state/se/pbmc_parse_with_splits_se.npy \\
#     sbatch --mem=768G --time=48:00:00 sbatch/third_party/encode_se.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/third_party/arc-state}"
if [[ ! -x "${STATE_ROOT}/.venv/bin/state" ]] && [[ ! -f "${STATE_ROOT}/pyproject.toml" ]]; then
  echo "ERROR: arc-state missing at ${STATE_ROOT}" >&2
  echo "Run: ./third_party/setup_arc_state.sh" >&2
  exit 1
fi

: "${MODEL_REPO:=arcinstitute/SE-600M}"
: "${DATA_REPO:=arcinstitute/Replogle-Nadig-Preprint}"
: "${DATA_REPO_TYPE:=dataset}"
: "${MODEL_DIR:=${REPO_ROOT}/data/arc_state/SE-600M}"
: "${DATA_DIR:=${REPO_ROOT}/data/arc_state/Replogle-Nadig-Preprint}"
: "${OUT_DIR:=${REPO_ROOT}/results_dir/arc_state/se}"
: "${INPUT_H5AD:=${REPO_ROOT}/data_dir/replogle.h5ad}"
: "${EMBED_KEY:=X_state}"
: "${BATCH_SIZE:=256}"
: "${SKIP_DOWNLOAD:=0}"
: "${DOWNLOAD_DATA:=0}"
: "${DRY_RUN:=0}"

mkdir -p "${MODEL_DIR}" "${OUT_DIR}"

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  echo "Ensuring SE-600M under ${MODEL_DIR}"
  # The venv's `huggingface-cli` shim is deprecated and exits non-zero, so drive
  # downloads through the Python API (resumes partial snapshots).
  "${STATE_ROOT}/.venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${MODEL_REPO}", local_dir="${MODEL_DIR}")
PY

  if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
    mkdir -p "${DATA_DIR}"
    echo "DOWNLOAD_DATA=1 — fetching ${DATA_REPO} -> ${DATA_DIR}"
    "${STATE_ROOT}/.venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${DATA_REPO}", repo_type="${DATA_REPO_TYPE}", local_dir="${DATA_DIR}")
PY
  fi
fi

# Resolve checkpoint (HF SE-600M ships se600m_epoch16.ckpt)
if [[ -z "${CHECKPOINT:-}" ]]; then
  for cand in se600m_epoch16.ckpt se600m_epoch15.ckpt; do
    if [[ -f "${MODEL_DIR}/${cand}" ]]; then
      CHECKPOINT="${MODEL_DIR}/${cand}"
      break
    fi
  done
  if [[ -z "${CHECKPOINT:-}" ]]; then
    mapfile -t _ckpts < <(find "${MODEL_DIR}" -maxdepth 2 -type f -name '*.ckpt' | sort)
    if [[ ${#_ckpts[@]} -eq 0 ]]; then
      echo "ERROR: no *.ckpt under ${MODEL_DIR}" >&2
      exit 1
    fi
    CHECKPOINT="${_ckpts[-1]}"
  fi
elif [[ "${CHECKPOINT}" != /* ]] && [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${MODEL_DIR}/${CHECKPOINT}"
fi

if [[ ! -f "${INPUT_H5AD}" ]]; then
  echo "ERROR: INPUT_H5AD not found: ${INPUT_H5AD}" >&2
  echo "Place the AnnData file at data_dir/replogle.h5ad or set INPUT_H5AD." >&2
  exit 1
fi

STEM="$(basename "${INPUT_H5AD}" .h5ad)"
: "${OUTPUT_H5AD:=${OUT_DIR}/${STEM}_se.h5ad}"

echo "STATE_ROOT=${STATE_ROOT}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "INPUT_H5AD=${INPUT_H5AD}"
echo "OUTPUT_H5AD=${OUTPUT_H5AD}"
echo "EMBED_KEY=${EMBED_KEY}"
echo "BATCH_SIZE=${BATCH_SIZE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — exiting before transform"
  exit 0
fi

cd "${STATE_ROOT}"
TRANSFORM_ARGS=(
  emb transform
  --model-folder "${MODEL_DIR}"
  --checkpoint "${CHECKPOINT}"
  --input "${INPUT_H5AD}"
  --output "${OUTPUT_H5AD}"
  --embed-key "${EMBED_KEY}"
)
if [[ -n "${BATCH_SIZE:-}" ]]; then
  TRANSFORM_ARGS+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -x "${STATE_ROOT}/.venv/bin/state" ]]; then
  "${STATE_ROOT}/.venv/bin/state" "${TRANSFORM_ARGS[@]}"
else
  uv run state "${TRANSFORM_ARGS[@]}"
fi

echo "Wrote ${OUTPUT_H5AD}"
