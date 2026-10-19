#!/bin/bash
#SBATCH --job-name=dl-scimilarity
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/dl_scimilarity_%j.log
#SBATCH --partition=YOUR_CPU_PARTITION

# Fetch the SCimilarity encoder from Zenodo record 10685499.
#
#   sbatch sbatch/third_party/download_scimilarity_model.sbatch
#
# The published artifact is a single ~30 GiB tarball whose bulk is the cell
# search / annotation index. We download it to disk with resume, then extract
# only encoder.ckpt, gene_order.tsv, layer_sizes.json, label_ints.csv.
# A piped stream died mid-transfer (curl 18); a file can be resumed.
#
# Env overrides:
#   MODEL_URL   — Zenodo file URL
#   DEST        — output folder (default: data/scimilarity/model)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

: "${MODEL_URL:=https://zenodo.org/api/records/10685499/files/model_v1.1.tar.gz/content}"
: "${DEST:=${REPO_ROOT}/data/scimilarity/model}"

WANTED=(encoder.ckpt gene_order.tsv layer_sizes.json label_ints.csv)

have_all() {
  local d="$1" f
  for f in "${WANTED[@]}"; do
    [[ -f "${d}/${f}" ]] || return 1
  done
  return 0
}

if have_all "${DEST}"; then
  echo "Encoder already present in ${DEST}; nothing to do."
  ls -la "${DEST}"
  exit 0
fi

TARBALL="${REPO_ROOT}/data/scimilarity/model_v1.1.tar.gz"
STAGE="$(mktemp -d "${REPO_ROOT}/data/scimilarity/.stage.XXXXXX")"
cleanup() {
  rm -rf "${STAGE}"
  if [[ "${KEEP_TARBALL:-0}" != "1" ]] && have_all "${DEST}"; then
    rm -f "${TARBALL}"
  fi
}
trap cleanup EXIT
mkdir -p "${DEST}" "$(dirname "${TARBALL}")"

echo "Downloading ${MODEL_URL} -> ${TARBALL} (resume enabled)"
curl -L --retry 20 --retry-delay 15 --retry-all-errors --continue-at - \
  --fail --progress-bar \
  -o "${TARBALL}" "${MODEL_URL}"

echo "Extracting only: ${WANTED[*]}"
tar -xzf "${TARBALL}" -C "${STAGE}" --wildcards --no-anchored \
  'encoder.ckpt' 'gene_order.tsv' 'layer_sizes.json' 'label_ints.csv'

CKPT="$(find "${STAGE}" -name encoder.ckpt -type f | head -n 1)"
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: encoder.ckpt not found in archive" >&2
  find "${STAGE}" -type f >&2
  exit 1
fi
SRC="$(dirname "${CKPT}")"

for f in "${WANTED[@]}"; do
  if [[ ! -f "${SRC}/${f}" ]]; then
    echo "ERROR: ${f} missing under ${SRC}" >&2
    find "${STAGE}" -type f >&2
    exit 1
  fi
  cp -f "${SRC}/${f}" "${DEST}/${f}"
done

echo "Wrote encoder to ${DEST}"
ls -la "${DEST}"
