"""Tests for MNIST KNN-stratified eval helpers."""

from __future__ import annotations

import numpy as np
import torch

from experiments.mnist.mnist_eval import compare_clouds, summarize_strata


def test_summarize_strata_empty() -> None:
    assert summarize_strata([])["n_strata"] == 0


def test_summarize_strata_means() -> None:
    rows = [
        {"mse_to_true": 0.1, "energy": 0.2, "mse_mean_vectors": 0.01},
        {"mse_to_true": 0.3, "energy": 0.4, "mse_mean_vectors": 0.02},
    ]
    s = summarize_strata(rows)
    assert s["n_strata"] == 2
    assert abs(s["mean_mse_to_true"] - 0.2) < 1e-6
    assert abs(s["median_energy"] - 0.3) < 1e-6


def test_compare_clouds_same_shape() -> None:
    x = torch.rand(16, 1, 28, 28)
    row = compare_clouds(x, x + 0.01, digit=3, seed=0, max_pairs=16)
    assert row["p"] == 3
    assert row["n_true"] == 16
    assert row["n_gen"] == 16
    assert row["mse_to_true"] > 0.0
    assert row["energy"] >= 0.0
