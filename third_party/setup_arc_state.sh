#!/usr/bin/env bash
# Clone / pin / sync ArcInstitute/state into third_party/arc-state (isolated UV env).
# Does NOT touch the repo root .venv (Python 3.13+).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_FILE="${REPO_ROOT}/third_party/arc-state.PIN.txt"
DEST="${REPO_ROOT}/third_party/arc-state"
REPO_URL="${ARC_STATE_REPO_URL:-https://github.com/ArcInstitute/state.git}"
PYTHON="${ARC_STATE_PYTHON:-3.12}"

if [[ ! -f "${PIN_FILE}" ]]; then
  echo "ERROR: missing pin file ${PIN_FILE}" >&2
  exit 1
fi
PIN="$(tr -d '[:space:]' < "${PIN_FILE}")"

mkdir -p "${REPO_ROOT}/third_party"

if [[ ! -d "${DEST}/.git" ]]; then
  echo "Cloning ${REPO_URL} -> ${DEST}"
  git clone "${REPO_URL}" "${DEST}"
fi

cd "${DEST}"
git fetch --tags origin
git checkout "${PIN}"
echo "Checked out $(git rev-parse HEAD) (want ${PIN})"

# Prefer a project-local UV cache if the sandbox/default cache is unwritable.
if [[ -z "${UV_CACHE_DIR:-}" ]] || [[ ! -w "${UV_CACHE_DIR}" ]]; then
  export UV_CACHE_DIR="${REPO_ROOT}/.uv-cache"
  mkdir -p "${UV_CACHE_DIR}"
fi

echo "uv sync --python ${PYTHON} (isolated .venv under ${DEST})"
uv sync --python "${PYTHON}"

echo
echo "OK. Try:"
echo "  cd ${DEST} && uv run state --help"
echo "  cd ${DEST} && uv run state emb transform --help"
