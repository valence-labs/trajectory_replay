"""Tests for in-training val ODE monitor summaries."""

from __future__ import annotations

from experiments.tx.metrics_summary_utils import summarize_val_ode_monitor


def test_summarize_val_ode_monitor_mean_and_median() -> None:
    held_out = frozenset({1})
    rows = [
        {"c": 1, "p": 0, "energy": 4.0, "mse_to_true": 0.4, "mse_mean_vectors": 0.04},
        {"c": 1, "p": 1, "energy": 2.0, "mse_to_true": 0.2, "mse_mean_vectors": 0.02},
        {"c": 0, "p": 0, "energy": 999.0, "mse_to_true": 9.99, "mse_mean_vectors": 0.99},
    ]
    out = summarize_val_ode_monitor(rows, held_out)
    assert out["n_val_strata_held_out"] == 2
    assert out["val_energy_median"] == 3.0
    assert out["val_energy_mean"] == 3.0
    assert abs(out["val_mse_to_true_median"] - 0.3) < 1e-9
    assert abs(out["val_mse_to_true_mean"] - 0.3) < 1e-9
    assert out["val_mse_mean_vectors_median"] == 0.03
    assert out["val_mse_mean_vectors_mean"] == 0.03


def test_summarize_val_ode_monitor_mean_sensitive_to_hard_strata() -> None:
    held_out = frozenset({0})
    rows = [
        {"c": 0, "p": 0, "energy": 1.0, "mse_to_true": 0.1, "mse_mean_vectors": 0.01},
        {"c": 0, "p": 1, "energy": 1.0, "mse_to_true": 0.1, "mse_mean_vectors": 0.01},
        {"c": 0, "p": 2, "energy": 10.0, "mse_to_true": 1.0, "mse_mean_vectors": 0.1},
    ]
    out = summarize_val_ode_monitor(rows, held_out)
    assert out["val_energy_median"] == 1.0
    assert out["val_energy_mean"] == 4.0
    assert out["val_mse_to_true_median"] == 0.1
    assert abs(out["val_mse_to_true_mean"] - (1.2 / 3)) < 1e-9
