#!/usr/bin/env python3
"""Audit and checksum the final PBMC and Replogle CL result sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
RESULT_FILES = (
    "config.json",
    "final_metrics.json",
    "cell_eval_minimal_stream_heldout.json",
)
MANIFEST_PATH = ROOT / "experiments/tx/tables/final_cl_results_manifest.json"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    pca_tag: str
    pca_dim: int
    suffix: str
    target_steps: int
    later_steps: int
    hidden_dim: int
    depth: int
    backbone_type: str
    decoder_name: str
    table: str
    plot_script: str
    plot_image: str


SPECS = (
    DatasetSpec(
        name="pbmc",
        pca_tag="pca100",
        pca_dim=100,
        suffix="final_res2048_d6",
        target_steps=300_000,
        later_steps=150_000,
        hidden_dim=2048,
        depth=6,
        backbone_type="residual_mlp",
        decoder_name="decoders_pca100_affine",
        table="experiments/tx/tables/pbmc_cl_pearson_pca100.tex",
        plot_script="experiments/tx/plot_pbmc_cl_fit_retention_heldout.py",
        plot_image="experiments/tx/tables/pbmc_cl_pca100_fit_retention_heldout.png",
    ),
    DatasetSpec(
        name="replogle",
        pca_tag="pca50",
        pca_dim=50,
        suffix="final_mlp2048_d4",
        target_steps=100_000,
        later_steps=50_000,
        hidden_dim=2048,
        depth=4,
        backbone_type="mlp",
        decoder_name="decoders_pca50_affine",
        table="experiments/tx/tables/replogle_cl_pearson_pca50.tex",
        plot_script="experiments/tx/plot_replogle_cl_fit_retention_heldout.py",
        plot_image="experiments/tx/tables/replogle_cl_pca50_fit_retention_heldout.png",
    ),
)

# Table label, run-name portion after the PCA tag.
METHODS = (
    (r"Naive ($vf{=}0$, $pert{=}0$, $ctx{=}0$)", "standard_embedding_p0c0"),
    (r"Naive ($vf{=}0$, $pert{=}1$, $ctx{=}1$)", "standard_embedding_p1c1"),
    (r"Naive ($vf{=}1$, $pert{=}0$, $ctx{=}0$)", "standard_p0c0"),
    (r"Naive ($vf{=}1$, $pert{=}0$, $ctx{=}1$)", "standard_p0c1"),
    (r"Naive ($vf{=}1$, $pert{=}1$, $ctx{=}0$)", "standard_p1c0"),
    (r"Naive ($vf{=}1$, $pert{=}1$, $ctx{=}1$)", "standard_p1c1"),
    (r"LoRA ($r{=}4$, $\alpha{=}8$)", "standard_lora_r4_p0c0"),
    (r"LoRA ($r{=}8$, $\alpha{=}16$)", "standard_lora_r8_p0c0"),
    (r"LoRA ($r{=}16$, $\alpha{=}32$)", "standard_lora_r16_p0c0"),
    (r"GFT (sigmoid, $vf{=}1$)", "gft_sigmoid_p0c0"),
    (r"GFT (linear, $vf{=}1$)", "gft_linear_p0c0"),
    (r"Two-stage (10\% new ctx $\to$ $vf{=}1$)", "standard_tsf0.1_p0c0"),
    (r"Two-stage (50\% new ctx $\to$ $vf{=}1$)", "standard_tsf0.5_p0c0"),
    (r"Real replay ($\lambda$-grid)", "real_grid"),
    (r"Real replay ($\lambda$-equal)", "real_equal"),
    (r"Generative replay (ODE, $\lambda$-grid)", "generative_ode_grid"),
    (r"Generative replay (ODE, $\lambda$-equal)", "generative_ode_equal"),
    (
        r"Generative replay (MeanFlow $N{=}1$, $\lambda$-grid)",
        "generative_mf_n1_grid",
    ),
    (
        r"Generative replay (MeanFlow $N{=}1$, $\lambda$-equal)",
        "generative_mf_n1_equal",
    ),
    (
        r"Generative replay (MeanFlow $N{=}4$, $\lambda$-grid)",
        "generative_mf_n4_grid",
    ),
    (
        r"Generative replay (MeanFlow $N{=}4$, $\lambda$-equal)",
        "generative_mf_n4_equal",
    ),
    (r"Trajectory replay (ODE, $\lambda$-grid)", "trajectory_ode_grid"),
    (r"Trajectory replay (ODE, $\lambda$-equal)", "trajectory_ode_equal"),
    (
        r"Trajectory replay (MeanFlow $N{=}1$, $\lambda$-grid)",
        "trajectory_mf_n1_grid",
    ),
    (
        r"Trajectory replay (MeanFlow $N{=}1$, $\lambda$-equal)",
        "trajectory_mf_n1_equal",
    ),
    (
        r"Trajectory replay (MeanFlow $N{=}4$, $\lambda$-grid)",
        "trajectory_mf_n4_grid",
    ),
    (
        r"Trajectory replay (MeanFlow $N{=}4$, $\lambda$-equal)",
        "trajectory_mf_n4_equal",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stem(base: str, pca_tag: str) -> str:
    parts = base.split("_")
    strategy = parts[0]
    if strategy == "real":
        return f"real_{pca_tag}_{parts[1]}"
    if strategy in {"generative", "trajectory"}:
        return f"{strategy}_{pca_tag}_{'_'.join(parts[1:-1])}_{parts[-1]}"
    return f"{strategy}_{pca_tag}_{'_'.join(parts[1:])}"


def displayed_table_label(label: str) -> str:
    return (
        label.replace(r"$\lambda$-grid", r"adaptive $\lambda$")
        .replace(r"$\lambda$-equal", r"equal-task $\lambda$")
        .replace(r"MeanFlow $N{=}1$", "MF-1")
        .replace(r"MeanFlow $N{=}4$", "MF-4")
    )


def run_path(spec: DatasetSpec, base: str, seed: int) -> Path:
    name = (
        f"{run_stem(base, spec.pca_tag)}"
        f"_st{spec.target_steps}_ft{spec.later_steps}_{spec.suffix}_s{seed}"
    )
    return ROOT / "results_dir/tx_results" / spec.name / name


def mean_overlap(track: dict[str, Any], include: Callable[[str], bool]) -> float:
    values = [
        float(result["overlap_at_N"])
        for key, payload in track["slices"].items()
        if include(key)
        for result in payload["perturbation_results"]
    ]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("CellEval slice has missing or non-finite overlap values")
    return mean(values)


def run_metrics(spec: DatasetSpec, run: Path) -> tuple[float, ...]:
    pearson = json.loads((run / "final_metrics.json").read_text())["final"]
    celleval = json.loads(
        (run / "cell_eval_minimal_stream_heldout.json").read_text()
    )["final"]
    if spec.name == "pbmc":
        later_donors = {"Donor1", "Donor4", "Donor9", "Donor12"}
        stream = pearson["stream"]["per_task"]
        heldout = pearson["heldout"]["per_task"]
        values = (
            mean(float(stream[str(task)]["pearson_delta"]) for task in range(1, 5)),
            float(stream["0"]["pearson_delta"]),
            mean(float(heldout[str(task)]["pearson_delta"]) for task in range(1, 5)),
            mean_overlap(
                celleval["stream"],
                lambda key: key.split("__", 1)[0] in later_donors,
            ),
            mean_overlap(
                celleval["stream"],
                lambda key: key.split("__", 1)[0] not in later_donors,
            ),
            mean_overlap(
                celleval["heldout"],
                lambda key: key.split("__", 1)[0] in later_donors,
            ),
        )
    else:
        values = (
            float(pearson["stream"]["per_task"]["1"]["pearson_delta"]),
            float(pearson["stream"]["per_task"]["0"]["pearson_delta"]),
            float(pearson["heldout"]["per_task"]["1"]["pearson_delta"]),
            mean_overlap(celleval["stream"], lambda key: "hepg2" in key.lower()),
            mean_overlap(celleval["stream"], lambda key: "hepg2" not in key.lower()),
            mean_overlap(celleval["heldout"], lambda key: "hepg2" in key.lower()),
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite reported aggregate in {run}")
    return values


def expected_config(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "dataset": spec.name,
        "representation": "pca",
        "pca_dim": spec.pca_dim,
        "target_steps": spec.target_steps,
        "later_target_steps": spec.later_steps,
        "hidden_dim": spec.hidden_dim,
        "depth": spec.depth,
        "backbone_type": spec.backbone_type,
        "batch_size": 1024,
        "lr": 1e-4,
        "later_lr": 0.0,
        "ode_step_size": 0.05,
        "eval_gen_per_stratum": 500,
        "lambda_gen_per_stratum": 200,
        "pred_control": "decoded",
        "flow_source": "noise",
        "replay_condition_sampling": "proportional",
        "trajectory_batch_sampling": "equal",
        "trajectory_replay_batch_multiplier": 1.0,
        "trajectory_replay_loss_multiplier": 1.0,
        "trajectory_time_sampling": "batch",
    }


def expected_method_config(base: str) -> dict[str, Any]:
    parts = base.split("_")
    strategy = parts[0]
    expected: dict[str, Any] = {
        "strategy": strategy,
        "finetune_train_old_pert_embeddings": False,
        "finetune_train_old_context_embeddings": False,
        "two_stage_1_epoch_fraction": None,
    }
    if strategy in {"real", "generative", "trajectory"}:
        expected["finetune_scope"] = "full"
        expected["lambda_mode"] = parts[-1]
        if "mf" in parts:
            expected["replay_backend"] = "mean_flow"
            expected["mean_flow_steps"] = int(parts[parts.index("mf") + 1][1:])
        else:
            expected["replay_backend"] = "ode"
            expected["mean_flow_steps"] = 1
        return expected

    expected["lambda_mode"] = "none"
    expected["replay_backend"] = "ode"
    expected["mean_flow_steps"] = 1
    if strategy == "gft":
        expected["finetune_scope"] = "full"
        expected["gft_beta_schedule"] = parts[1]
        return expected

    expected["finetune_scope"] = (
        "embedding"
        if "embedding" in parts
        else "lora"
        if "lora" in parts
        else "full"
    )
    flags = re.search(r"_p([01])c([01])$", base)
    if flags is not None:
        expected["finetune_train_old_pert_embeddings"] = flags.group(1) == "1"
        expected["finetune_train_old_context_embeddings"] = flags.group(2) == "1"
    if "lora" in parts:
        expected["lora_rank"] = int(parts[parts.index("lora") + 1][1:])
    if "tsf0.1" in parts:
        expected["two_stage_1_epoch_fraction"] = 0.1
    elif "tsf0.5" in parts:
        expected["two_stage_1_epoch_fraction"] = 0.5
    return expected


def audit_dataset(spec: DatasetSpec) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    root = ROOT / "results_dir/tx_results" / spec.name
    expected_runs = [
        run_path(spec, base, seed) for _, base in METHODS for seed in SEEDS
    ]
    actual_runs = set(root.glob(f"*_{spec.suffix}_s*"))
    if set(expected_runs) != actual_runs:
        errors.append(
            f"{spec.name}: run-set mismatch "
            f"(missing={len(set(expected_runs) - actual_runs)}, "
            f"extra={len(actual_runs - set(expected_runs))})"
        )

    aggregates: dict[str, list[tuple[float, ...]]] = {
        base: [] for _, base in METHODS
    }
    manifest_runs: list[dict[str, Any]] = []
    config_want = expected_config(spec)
    for label, base in METHODS:
        for seed in SEEDS:
            run = run_path(spec, base, seed)
            missing = [name for name in RESULT_FILES if not (run / name).is_file()]
            if missing:
                errors.append(f"{run.name}: missing {missing}")
                continue
            config = json.loads((run / "config.json").read_text())
            for key, want in config_want.items():
                if config.get(key) != want:
                    errors.append(
                        f"{run.name}: {key}={config.get(key)!r}, expected {want!r}"
                    )
            for key, want in expected_method_config(base).items():
                if config.get(key) != want:
                    errors.append(
                        f"{run.name}: {key}={config.get(key)!r}, expected {want!r}"
                    )
            if config.get("seed") != seed:
                errors.append(f"{run.name}: seed={config.get('seed')}, expected {seed}")
            if Path(config.get("decoder_dir", "")).name != spec.decoder_name:
                errors.append(f"{run.name}: wrong decoder bank")

            final = json.loads((run / "final_metrics.json").read_text())
            celleval = json.loads(
                (run / "cell_eval_minimal_stream_heldout.json").read_text()
            )
            if set(final.get("final", {})) != {"stream", "heldout"}:
                errors.append(f"{run.name}: wrong Pearson tracks")
            if set(celleval.get("final", {})) != {"stream", "heldout"}:
                errors.append(f"{run.name}: wrong CellEval tracks")
            if final.get("eval_provenance", {}).get("gen_per_stratum") != 500:
                errors.append(f"{run.name}: Pearson provenance K is not 500")
            if celleval.get("eval_provenance", {}).get("gen_per_stratum") != 500:
                errors.append(f"{run.name}: CellEval provenance K is not 500")
            try:
                aggregates[base].append(run_metrics(spec, run))
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{run.name}: {exc}")

            manifest_runs.append(
                {
                    "method": label,
                    "seed": seed,
                    "path": str(run.relative_to(ROOT)),
                    "files": {
                        name: sha256(run / name)
                        for name in RESULT_FILES
                    },
                }
            )

    expected_rows: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for label, base in METHODS:
        rows = aggregates[base]
        if len(rows) != len(SEEDS):
            errors.append(f"{spec.name}/{base}: expected 3 aggregate rows")
            continue
        expected_rows.append(
            (
                label,
                tuple(
                    (mean(column), stdev(column) / math.sqrt(len(SEEDS)))
                    for column in zip(*rows)
                ),
            )
        )

    table_text = (ROOT / spec.table).read_text()
    number_pattern = re.compile(
        r"\$(?:\\mathbf\{)?([0-9]+\.[0-9]+)(?:\})?\$ "
        r"\{\\scriptsize \$\(\\pm\\,([0-9]+\.[0-9]+)\)\$\}"
    )
    for label, values in expected_rows:
        table_label = displayed_table_label(label)
        start = table_text.find(table_label)
        if start < 0:
            errors.append(f"{spec.name}: missing table row {table_label}")
            continue
        end = table_text.find(r"\\", start)
        got = number_pattern.findall(table_text[start:end])
        want = [(f"{avg:.3f}", f"{sem:.3f}") for avg, sem in values]
        if got != want:
            errors.append(f"{spec.name}: table mismatch for {label}")

    namespace = runpy.run_path(str(ROOT / spec.plot_script))
    points = namespace["POINTS"]
    got_points = Counter(
        (round(point.fit, 3), round(point.retention, 3), round(point.heldout, 3))
        for point in points
    )
    want_points = Counter(
        tuple(round(values[index][0], 3) for index in range(3))
        for _, values in expected_rows
    )
    if got_points != want_points:
        errors.append(f"{spec.name}: plot points do not match result aggregates")

    return manifest_runs, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest and args.verify_manifest:
        parser.error("--write-manifest and --verify-manifest are mutually exclusive")

    all_runs: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for spec in SPECS:
        runs, dataset_errors = audit_dataset(spec)
        all_runs[spec.name] = runs
        errors.extend(dataset_errors)
        print(
            f"{spec.name}: {len(runs)}/{len(METHODS) * len(SEEDS)} runs audited"
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"Final CL audit failed with {len(errors)} error(s)")

    print("tables: verified against raw three-seed aggregates")
    print("plots: verified against table aggregates")
    if args.verify_manifest:
        manifest = json.loads(MANIFEST_PATH.read_text())
        integrity_errors: list[str] = []
        checked = 0

        def verify(path_text: str, want: str) -> None:
            nonlocal checked
            path = ROOT / path_text
            if not path.is_file():
                integrity_errors.append(f"missing checksummed file: {path_text}")
            elif sha256(path) != want:
                integrity_errors.append(f"checksum mismatch: {path_text}")
            checked += 1

        for path_text, checksum in manifest["code"].items():
            verify(path_text, checksum)
        for dataset_artifacts in manifest["artifacts"].values():
            for artifact in dataset_artifacts.values():
                verify(artifact["path"], artifact["sha256"])
        for dataset_runs in manifest["runs"].values():
            for run in dataset_runs:
                for name, checksum in run["files"].items():
                    verify(f"{run['path']}/{name}", checksum)
        if integrity_errors:
            for error in integrity_errors:
                print(f"ERROR: {error}")
            raise SystemExit(
                f"Final CL manifest verification failed with "
                f"{len(integrity_errors)} error(s)"
            )
        print(f"manifest: verified {checked} checksummed files")
    if args.write_manifest:
        tracked_artifacts = {
            spec.name: {
                "table": {
                    "path": spec.table,
                    "sha256": sha256(ROOT / spec.table),
                },
                "plot_script": {
                    "path": spec.plot_script,
                    "sha256": sha256(ROOT / spec.plot_script),
                },
                "plot_image": {
                    "path": spec.plot_image,
                    "sha256": sha256(ROOT / spec.plot_image),
                },
                "decoder_manifest": {
                    "path": str(
                        Path("results_dir/tx_results")
                        / spec.name
                        / spec.decoder_name
                        / "manifest.json"
                    ),
                    "sha256": sha256(
                        ROOT
                        / "results_dir/tx_results"
                        / spec.name
                        / spec.decoder_name
                        / "manifest.json"
                    ),
                },
            }
            for spec in SPECS
        }
        code_paths = (
            "experiments/tx/train_cl.py",
            "experiments/tx/eval_cl_celleval.py",
            "experiments/tx/eval_cl_heldout.py",
            "experiments/tx/audit_final_cl_results.py",
            "src/trajectory_replay/training/flow_matching.py",
            "src/trajectory_replay/training/mean_flow_cl.py",
            "src/trajectory_replay/eval/tx_cl_pearson.py",
            "src/trajectory_replay/data/tx_cl_tasks.py",
            "src/trajectory_replay/model/backbones.py",
            "src/trajectory_replay/model/flow_matching.py",
            "src/trajectory_replay/model/lora.py",
            "sbatch/tx/train_cl.sbatch",
            "sbatch/tx/eval_cl_celleval.sbatch",
            "sbatch/tx/submit_final_pbmc_replay.sh",
            "sbatch/tx/submit_final_pbmc_baselines.sh",
            "sbatch/tx/submit_final_replogle_replay.sh",
            "sbatch/tx/submit_final_replogle_baselines.sh",
        )
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git_head,
            "protocol": {
                "seeds": list(SEEDS),
                "final_generated_cells_per_non_control_stratum": 500,
                "lambda_grid_selection_generated_cells_per_stratum": 200,
                "predicted_controls": "decoded observed control latents",
            },
            "code": {
                path: sha256(ROOT / path)
                for path in code_paths
            },
            "artifacts": tracked_artifacts,
            "runs": all_runs,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    print("FINAL CL AUDIT PASSED")


if __name__ == "__main__":
    main()
