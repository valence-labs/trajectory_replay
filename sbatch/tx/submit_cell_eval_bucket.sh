#!/usr/bin/env bash
# Submit Cell-Eval for every run listed in a methods comparison manifest.
#
#   TX_RESULTS_BUCKET=pbmc_split3_state ./sbatch/tx/submit_cell_eval_bucket.sh
#   TX_RESULTS_BUCKET=replogle_split2_hepg2 ./sbatch/tx/submit_cell_eval_bucket.sh
#
#   DRY_RUN=1 ./sbatch/tx/submit_cell_eval_bucket.sh
#   RUN_LOCAL=1 RUN_PATH=pbmc_split3_pretrain TX_RESULTS_BUCKET=pbmc_split3_state ./sbatch/tx/submit_cell_eval_bucket.sh
#   SKIP_DONE=0 ./sbatch/tx/submit_cell_eval_bucket.sh   # re-run even if metrics.json has cell_eval_*
#   BATCH_SIZE=10 TX_RESULTS_BUCKET=replogle_split2_hepg2 ./sbatch/tx/submit_cell_eval_bucket.sh
#     Submit at most BATCH_SIZE Slurm jobs, wait for the batch to finish, then submit the next.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tx_results_bucket.env.sh
source "${SCRIPT_DIR}/tx_results_bucket.env.sh"
SBATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="$(cd "${SBATCH_ROOT}/.." && pwd)"
cd "${SIM_ROOT}"

mkdir -p "${SBATCH_ROOT}/logs"

export PROFILE="${PROFILE:-full}"
export CELL_EVAL_EXPORT_PROTOCOL="${CELL_EVAL_EXPORT_PROTOCOL:-10}"
export USE_OBSERVED_CONTROLS_ONLY="${USE_OBSERVED_CONTROLS_ONLY:-1}"
export USE_RAW_X="${USE_RAW_X:-1}"
export EVAL_PHASES="${EVAL_PHASES:-auto}"
export SBATCH_MEM="${SBATCH_MEM:-256G}"
export SBATCH_TIME="${SBATCH_TIME:-6:00:00}"

DRY_RUN="${DRY_RUN:-0}"
RUN_LOCAL="${RUN_LOCAL:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
BATCH_SIZE="${BATCH_SIZE:-0}"
TX_RESULTS_BUCKET="${TX_RESULTS_BUCKET:-tx_results/pbmc}"
case "${TX_RESULTS_BUCKET}" in
  tx_results/pbmc) MANIFEST="${MANIFEST:-configs/pbmc_split3_state_methods_comparison_all.tsv}" ;;
  tx_results/replogle) MANIFEST="${MANIFEST:-configs/replogle_split2_hepg2_methods_comparison_all.tsv}" ;;
  tx_results_old/pbmc_split3_state|pbmc_split3_state)
    MANIFEST="${MANIFEST:-configs/pbmc_split3_state_methods_comparison_all.tsv}"
    ;;
  tx_results_old/replogle_split2_hepg2|replogle_split2_hepg2)
    MANIFEST="${MANIFEST:-configs/replogle_split2_hepg2_methods_comparison_all.tsv}"
    ;;
  *) MANIFEST="${MANIFEST:-configs/${TX_RESULTS_BUCKET}_methods_comparison_all.tsv}" ;;
esac
SBATCH_SCRIPT="${SCRIPT_DIR}/run_cell_eval.sbatch"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

export SIM_ROOT
export TX_RESULTS_BUCKET
export RESULTS_DIR="./results_dir/${TX_RESULTS_BUCKET}"

_run_one() {
  local run_path="$1"
  local label="$2"
  local stage="${3:-finetune}"
  local run_dir="${RESULTS_DIR}/${run_path}"

  if [[ ! -f "${run_dir}/checkpoint.pt" ]]; then
    echo "skip (no checkpoint): ${run_path}" >&2
    return 2
  fi

  if [[ "${SKIP_DONE}" == "1" && -f "${run_dir}/metrics.json" ]]; then
    if uv run python - "${run_dir}" <<'PY'
import json
import os
import sys
from pathlib import Path

repo = Path(os.environ["SIM_ROOT"])
run_dir = Path(sys.argv[1])
metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
margs = metrics.get("args") or {}
h5ad = margs.get("h5ad_path")
if h5ad is None:
    from trajectory_replay.data.tx_dataset import default_h5ad_path
    h5ad = default_h5ad_path()
h5ad = Path(h5ad)
if not h5ad.is_file():
    sys.exit(1)
from experiments.tx.tx_common import cell_eval_run_is_complete

want_obs = os.environ.get("USE_OBSERVED_CONTROLS_ONLY", "1").lower() not in ("0", "false", "no")
want_raw = os.environ.get("USE_RAW_X", "1").lower() not in ("0", "false", "no")
agg_env = os.environ.get("AGGREGATE_COVARIATE")
enc_meta = metrics.get("encoding") or {}
ctx_col = str(margs.get("context_col") or "donor")
cov_order = tuple(enc_meta.get("covariate_col_order") or ())
if agg_env is not None and str(agg_env).strip().lower() in ("none", "null", ""):
    aggregate_covariate = None
elif agg_env:
    aggregate_covariate = str(agg_env).strip()
else:
    from experiments.tx.cell_eval_export import default_aggregate_covariate

    aggregate_covariate = default_aggregate_covariate(
        context_col=ctx_col,
        covariate_col_order=cov_order,
    )
ok = cell_eval_run_is_complete(
    metrics,
    h5ad,
    str(margs.get("split_col") or "split"),
    profile=os.environ.get("PROFILE", "full"),
    export_protocol_min=int(os.environ.get("CELL_EVAL_EXPORT_PROTOCOL", "10")),
    use_observed_controls_only=want_obs,
    use_raw_x=want_raw,
    eval_phases=os.environ.get("EVAL_PHASES", "auto"),
    aggregate_covariate=aggregate_covariate,
)
sys.exit(0 if ok else 1)
PY
    then
      echo "skip (cell_eval complete ${PROFILE:-full} EVAL_PHASES=${EVAL_PHASES:-auto}): ${run_path}" >&2
      return 2
    fi
  fi

  if [[ "${RUN_LOCAL}" == "1" ]]; then
    echo "local cell-eval: ${run_path}"
    uv sync --extra cell-eval
    uv run python -c "from cell_eval import MetricsEvaluator"
    local_args=(experiments/tx/run_cell_eval.py "${run_path}" --device "${DEVICE:-cuda}")
    if [[ -n "${EVAL_PHASES:-}" ]]; then
      local_args+=(--eval-phases "${EVAL_PHASES}")
    fi
    if [[ "${USE_OBSERVED_CONTROLS_ONLY}" == "0" ]]; then
      local_args+=(--no-use-observed-controls-only)
    fi
    if [[ "${USE_RAW_X}" == "0" ]]; then
      local_args+=(--no-use-raw-x)
    fi
    if [[ -n "${AGGREGATE_COVARIATE:-}" ]]; then
      local_args+=(--aggregate-covariate "${AGGREGATE_COVARIATE}")
    fi
    if [[ "${SKIP_EXISTING_PHASES:-0}" == "1" ]]; then
      local_args+=(--skip-existing-phases)
    fi
    uv run python "${local_args[@]}"
    return 0
  fi

  local log_file="${SBATCH_ROOT}/logs/tx_cell_eval_${TX_RESULTS_BUCKET}_${label//\//_}_%j.log"
  local export_cmd="ALL,SIM_ROOT=${SIM_ROOT},TX_RESULTS_BUCKET=${TX_RESULTS_BUCKET},RESULTS_DIR=${RESULTS_DIR},RUN_PATH=${run_path}"
  export_cmd+=",DEVICE=${DEVICE:-cuda},PROFILE=${PROFILE:-full},CELL_EVAL_EXPORT_PROTOCOL=${CELL_EVAL_EXPORT_PROTOCOL:-10},USE_OBSERVED_CONTROLS_ONLY=${USE_OBSERVED_CONTROLS_ONLY:-1},USE_RAW_X=${USE_RAW_X:-1},SKIP_EXISTING_PHASES=${SKIP_EXISTING_PHASES:-0},EVAL_PHASES=${EVAL_PHASES:-auto},RUN_STAGE=${stage}"
  if [[ -n "${AGGREGATE_COVARIATE:-}" ]]; then
    export_cmd+=",AGGREGATE_COVARIATE=${AGGREGATE_COVARIATE}"
  fi
  if [[ -n "${CONTEXT:-}" ]]; then
    export_cmd+=",CONTEXT=${CONTEXT}"
  fi

  local sbatch_cmd=(
    sbatch
    --export="${export_cmd}"
    --job-name="tx-ce-${label:0:20}"
    --mem="${SBATCH_MEM}"
    --time="${SBATCH_TIME}"
    --output="${log_file}"
    "${SBATCH_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${sbatch_cmd[*]}"
    return 0
  fi

  local sbatch_out job_id
  sbatch_out="$("${sbatch_cmd[@]}")"
  job_id="${sbatch_out##* }"
  echo "submitted: ${run_path} (job ${job_id})" >&2
  printf '%s\n' "${job_id}"
  return 0
}

_wait_for_jobs() {
  local -a job_ids=("$@")
  if ((${#job_ids[@]} == 0)); then
    return 0
  fi
  echo "waiting for batch jobs: ${job_ids[*]}"
  while true; do
    local pending=0
    for job_id in "${job_ids[@]}"; do
      if squeue -j "${job_id}" -h 2>/dev/null | grep -q .; then
        pending=$((pending + 1))
      fi
    done
    if ((pending == 0)); then
      break
    fi
    sleep 30
  done
  echo "batch finished (${#job_ids[@]} jobs)"
}

if [[ -n "${RUN_PATH:-}" ]]; then
  _run_one "${RUN_PATH}" "${RUN_PATH##*/}" "${RUN_STAGE:-finetune}"
  exit 0
fi

n_submitted=0
n_skipped=0
batch_idx=0
declare -a pending_runs=()
declare -a pending_labels=()
declare -a pending_stages=()

while IFS=$'\t' read -r label run_path stage || [[ -n "${label:-}" ]]; do
  [[ -z "${label:-}" || "${label}" == "label" || "${label}" =~ ^# ]] && continue
  pending_runs+=("${run_path}")
  pending_labels+=("${label}")
  pending_stages+=("${stage:-finetune}")
done < "${MANIFEST}"

if [[ "${BATCH_SIZE}" != "0" && "${RUN_LOCAL}" != "1" && -z "${RUN_PATH:-}" ]]; then
  echo "batched submit: BATCH_SIZE=${BATCH_SIZE} bucket=${TX_RESULTS_BUCKET} manifest=${MANIFEST}"
  idx=0
  while (( idx < ${#pending_runs[@]} )); do
    batch_idx=$((batch_idx + 1))
    declare -a job_ids=()
    submitted_in_batch=0
    echo "=== batch ${batch_idx} (queue position ${idx}/${#pending_runs[@]}) ==="
    while (( idx < ${#pending_runs[@]} && submitted_in_batch < BATCH_SIZE )); do
      run_path="${pending_runs[$idx]}"
      label="${pending_labels[$idx]}"
      stage="${pending_stages[$idx]}"
      idx=$((idx + 1))
      if [[ "${DRY_RUN}" == "1" ]]; then
        if _run_one "${run_path}" "${label}" "${stage}"; then
          submitted_in_batch=$((submitted_in_batch + 1))
          n_submitted=$((n_submitted + 1))
        else
          rc=$?
          if [[ "${rc}" -eq 2 ]]; then
            n_skipped=$((n_skipped + 1))
          fi
        fi
        continue
      fi
      if job_id="$(_run_one "${run_path}" "${label}" "${stage}")"; then
        job_ids+=("${job_id}")
        submitted_in_batch=$((submitted_in_batch + 1))
        n_submitted=$((n_submitted + 1))
      else
        rc=$?
        if [[ "${rc}" -eq 2 ]]; then
          n_skipped=$((n_skipped + 1))
        else
          echo "ERROR: failed to submit ${run_path} (rc=${rc})" >&2
          exit 1
        fi
      fi
    done
    if [[ "${DRY_RUN}" != "1" ]]; then
      _wait_for_jobs "${job_ids[@]}"
    fi
  done
else
  for i in "${!pending_runs[@]}"; do
    if _run_one "${pending_runs[$i]}" "${pending_labels[$i]}" "${pending_stages[$i]}"; then
      n_submitted=$((n_submitted + 1))
    else
      rc=$?
      if [[ "${rc}" -eq 2 ]]; then
        n_skipped=$((n_skipped + 1))
      fi
    fi
  done
fi

echo "manifest=${MANIFEST} bucket=${TX_RESULTS_BUCKET} submitted=${n_submitted} skipped=${n_skipped}"
