"""Tests for train/eval phase guards and OOD phase resolution."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

from experiments.tx.tx_common import (  # noqa: E402
    assert_grad_safe_phases,
    FINETUNE_GRAD_PHASES,
    FORBIDDEN_IN_FINETUNE_GRAD,
    FORBIDDEN_IN_PRETRAIN_GRAD,
    PRETRAIN_GRAD_PHASES,
    resolve_eval_ood_phases,
)


def test_pretrain_rejects_finetune_phase() -> None:
    try:
        assert_grad_safe_phases(
            "Pretrain",
            ("train", "finetune"),
            allowed=PRETRAIN_GRAD_PHASES,
            forbidden_extra=FORBIDDEN_IN_PRETRAIN_GRAD,
        )
    except ValueError as exc:
        assert "gradients forbidden" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_finetune_rejects_train_phase() -> None:
    try:
        assert_grad_safe_phases(
            "Finetune training",
            ("finetune", "train"),
            allowed=FINETUNE_GRAD_PHASES,
            forbidden_extra=FORBIDDEN_IN_FINETUNE_GRAD,
        )
    except ValueError as exc:
        assert "gradients forbidden" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_ood_phases_explicit() -> None:
    h5ad = _REPO / "data_dir/pbmc_parse_with_splits.h5ad"
    if not h5ad.is_file():
        return
    phases = resolve_eval_ood_phases(
        h5ad,
        "split",
        eval_ood_phases="val,test",
        no_eval_ood=False,
        legacy_holdout_phase=None,
    )
    assert phases == ("val", "test"), phases


def test_resolve_ood_phases_disabled() -> None:
    h5ad = _REPO / "data_dir/pbmc_parse_with_splits.h5ad"
    if not h5ad.is_file():
        return
    assert (
        resolve_eval_ood_phases(
            h5ad,
            "split",
            eval_ood_phases="val,test",
            no_eval_ood=True,
            legacy_holdout_phase=None,
        )
        == ()
    )


if __name__ == "__main__":
    test_pretrain_rejects_finetune_phase()
    test_finetune_rejects_train_phase()
    test_resolve_ood_phases_explicit()
    test_resolve_ood_phases_disabled()
    print("test_tx_eval_phases: ok")
