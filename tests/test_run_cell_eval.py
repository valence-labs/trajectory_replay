"""Tests for run_cell_eval CLI/metrics arg merging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "tx"))

from run_cell_eval import _merge_run_args
from experiments.tx.tx_common import resolve_cell_eval_phases


def test_merge_run_args_keeps_cli_only_fields_when_missing_from_metrics() -> None:
    cli = argparse.Namespace(
        run_dir="some/run",
        context=None,
        profile="vcc",
        out_subdir="cell_eval",
        num_threads=-1,
        save_h5ad=False,
        no_metrics_update=False,
        device="cuda",
        eval_phases=None,
        pert_col=None,
        use_observed_controls_only=True,
        use_raw_x=True,
    )
    metrics = {
        "args": {
            "h5ad_path": "data_dir/foo.h5ad",
            "obsm_key": "X_pca",
            "split_col": "split",
            "context_col": "donor",
            "perturbation_col": "cytokine",
        },
        "eval_ood_phases": ["val", "test"],
    }

    args = _merge_run_args(cli, metrics)

    assert args.use_raw_x is True
    assert args.context is None
    assert args.split_col == "split"
    assert args.profile == "vcc"
    assert args.out_subdir == "cell_eval"
    assert args.h5ad_path == Path("data_dir/foo.h5ad")
    assert getattr(args, "eval_phases", None) is None
    assert args.pert_col == "cytokine"


def test_merge_run_args_keeps_metrics_split_col_over_cli_default() -> None:
    cli = argparse.Namespace(
        run_dir="some/run",
        split_col="split1",
        context_col="donor",
        perturbation_col="cytokine",
        h5ad_path=None,
        context=None,
        profile="vcc",
        out_subdir="cell_eval",
        num_threads=-1,
        save_h5ad=False,
        no_metrics_update=False,
        device="cuda",
        eval_phases=None,
        pert_col=None,
        use_observed_controls_only=True,
        use_raw_x=True,
    )
    metrics = {
        "args": {
            "h5ad_path": "data_dir/replogle_with_splits.h5ad",
            "split_col": "split2",
            "context_col": "cell_line",
            "perturbation_col": "gene",
        },
        "eval_ood_phases": ["val", "test"],
    }

    args = _merge_run_args(cli, metrics)

    assert args.split_col == "split2"
    assert args.context_col == "cell_line"
    assert args.perturbation_col == "gene"


def test_merge_run_args_cli_overrides_metrics() -> None:
    cli = argparse.Namespace(
        run_dir="some/run",
        context="A,B",
        profile="minimal",
        out_subdir="cell_eval",
        num_threads=4,
        save_h5ad=False,
        no_metrics_update=False,
        device="cpu",
        eval_phases="test",
        pert_col="target",
        use_observed_controls_only=True,
        use_raw_x=True,
    )
    metrics = {
        "args": {
            "h5ad_path": "data_dir/foo.h5ad",
            "device": "cuda",
            "perturbation_col": "cytokine",
        },
    }

    args = _merge_run_args(cli, metrics)

    assert args.context == "A,B"
    assert args.profile == "minimal"
    assert args.device == "cpu"
    assert args.eval_phases == "test"
    assert args.pert_col == "target"
    assert args.use_observed_controls_only is True


def test_merge_run_args_cli_overrides_use_observed_controls_only() -> None:
    cli = argparse.Namespace(
        run_dir="some/run",
        context=None,
        profile="minimal",
        out_subdir="cell_eval",
        num_threads=-1,
        save_h5ad=False,
        no_metrics_update=False,
        device="cpu",
        eval_phases=None,
        pert_col=None,
        use_observed_controls_only=False,
        use_raw_x=True,
    )
    metrics = {"args": {"perturbation_col": "cytokine"}}

    args = _merge_run_args(cli, metrics)

    assert args.use_observed_controls_only is False
    assert args.use_raw_x is True


def test_merge_run_args_cli_overrides_use_raw_x_off() -> None:
    cli = argparse.Namespace(
        run_dir="some/run",
        context=None,
        profile="minimal",
        out_subdir="cell_eval",
        num_threads=-1,
        save_h5ad=False,
        no_metrics_update=False,
        device="cpu",
        eval_phases=None,
        pert_col=None,
        use_observed_controls_only=True,
        use_raw_x=False,
    )
    metrics = {"args": {"perturbation_col": "cytokine"}}

    args = _merge_run_args(cli, metrics)

    assert args.use_raw_x is False


def test_merge_run_args_defaults_use_raw_x_when_missing() -> None:
    cli = argparse.Namespace(
        run_dir="some/run",
        context=None,
        profile="minimal",
        out_subdir="cell_eval",
        num_threads=-1,
        save_h5ad=False,
        no_metrics_update=False,
        device="cpu",
        eval_phases=None,
        pert_col=None,
        use_observed_controls_only=True,
        use_raw_x=True,
    )
    metrics = {"args": {"perturbation_col": "cytokine"}}

    args = _merge_run_args(cli, metrics)

    assert args.use_raw_x is True


def test_resolve_cell_eval_phases_pretrain_replogle_auto_test() -> None:
    h5ad = Path(__file__).resolve().parents[1] / "data_dir/replogle_with_splits.h5ad"
    if not h5ad.is_file():
        return
    metrics = {
        "stage": "pretrain",
        "args": {"split_col": "split2", "train_phase": "train"},
    }
    phases = resolve_cell_eval_phases(
        metrics,
        h5ad,
        "split2",
        cli_eval_phases=None,
    )
    assert phases == ("test",), phases


def test_resolve_cell_eval_phases_finetune_uses_eval_ood_phases() -> None:
    metrics = {"eval_ood_phases": ["test"]}
    phases = resolve_cell_eval_phases(
        metrics,
        Path("/nonexistent.h5ad"),
        "split2",
        cli_eval_phases=None,
    )
    assert phases == ("test",)


def test_resolve_cell_eval_phases_auto_pretrain_train_only() -> None:
    from experiments.tx.tx_common import resolve_cell_eval_phases_auto

    h5ad = Path(__file__).resolve().parents[1] / "data_dir/replogle_with_splits.h5ad"
    if not h5ad.is_file():
        return
    metrics = {
        "stage": "pretrain",
        "args": {"train_phase": "train", "split_col": "split2"},
    }
    phases = resolve_cell_eval_phases_auto(metrics, h5ad, "split2")
    assert phases == ("train",)


def test_resolve_cell_eval_phases_auto_finetune_includes_train_and_test() -> None:
    from experiments.tx.tx_common import resolve_cell_eval_phases_auto

    h5ad = Path(__file__).resolve().parents[1] / "data_dir/replogle_with_splits.h5ad"
    if not h5ad.is_file():
        return
    metrics = {
        "stage": "finetune",
        "eval_ood_phases": ["test"],
        "args": {"train_phase": "train", "split_col": "split2"},
    }
    phases = resolve_cell_eval_phases_auto(metrics, h5ad, "split2")
    assert "train" in phases
    assert "test" in phases
