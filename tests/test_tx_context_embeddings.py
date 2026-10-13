"""Tests for full-table context embeddings (tx pretrain / finetune)."""

from __future__ import annotations

from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
from trajectory_replay.data.tx_dataset import build_encoding
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from experiments.tx.tx_common import (
    assert_tx_checkpoint_matches_encoding,
    resolve_context_label_indices,
)


def test_resolve_context_label_indices_single_and_list() -> None:
    enc = build_encoding(
        ["Donor1", "Donor2", "Donor8"],
        ["PBS", "IL-7"],
        control_perturbation="PBS",
    )
    one_rows, one_labels = resolve_context_label_indices(enc, "Donor8")
    assert one_rows == frozenset({enc.context_to_idx["Donor8"]})
    assert one_labels == ("Donor8",)
    multi_rows, multi_labels = resolve_context_label_indices(
        enc, "Donor1,Donor2"
    )
    assert multi_rows == frozenset({0, 1})
    assert multi_labels == ("Donor1", "Donor2")


def test_mask_embedding_rows_grad_zeros_non_train_rows() -> None:
    emb = torch.nn.Embedding(4, 3)
    emb.weight.grad = torch.ones_like(emb.weight)
    ConditionalFlowMatching.mask_embedding_rows_grad(emb.weight, frozenset({1, 3}))
    assert emb.weight.grad[0].abs().sum() == 0
    assert emb.weight.grad[2].abs().sum() == 0
    assert emb.weight.grad[1].abs().sum() > 0
    assert emb.weight.grad[3].abs().sum() > 0


def test_assert_tx_checkpoint_matches_encoding_rejects_legacy() -> None:
    enc = build_encoding(["Donor1", "Donor2"], ["PBS"], control_perturbation="PBS")
    model = ConditionalFlowMatching(
        data_dim=2,
        n_contexts=1,
        n_perturbations=1,
        cond_emb_dim=4,
        time_emb_dim=4,
        hidden_dim=8,
        depth=2,
    )
    try:
        assert_tx_checkpoint_matches_encoding(model, enc, stage="pretrain")
    except ValueError as exc:
        assert "Legacy" in str(exc)
    else:
        raise AssertionError("expected ValueError for legacy checkpoint")


def test_assert_tx_checkpoint_matches_encoding_accepts_full_table() -> None:
    enc = build_encoding(["Donor1", "Donor2"], ["PBS"], control_perturbation="PBS")
    model = ConditionalFlowMatching(
        data_dim=2,
        n_contexts=2,
        n_perturbations=1,
        cond_emb_dim=4,
        time_emb_dim=4,
        hidden_dim=8,
        depth=2,
    )
    assert_tx_checkpoint_matches_encoding(model, enc, stage="pretrain")


if __name__ == "__main__":
    test_resolve_context_label_indices_single_and_list()
    test_mask_embedding_rows_grad_zeros_non_train_rows()
    test_assert_tx_checkpoint_matches_encoding_rejects_legacy()
    test_assert_tx_checkpoint_matches_encoding_accepts_full_table()
    print("test_tx_context_embeddings: ok")
