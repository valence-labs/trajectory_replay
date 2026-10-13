from __future__ import annotations

import numpy as np

from experiments.sim_sequential.common import (
    build_dataset_bundle,
    make_clockwise_ring_means,
)
from experiments.sim_sequential.metrics import (
    categorical_kl,
    evaluate_sequential_samples,
    update_policy_weights,
)


def test_ring_starts_at_top_and_runs_clockwise() -> None:
    means = make_clockwise_ring_means(n_tasks=5, dim=2, radius=3.0)

    np.testing.assert_allclose(means[0], [0.0, 3.0], atol=1e-12)
    assert means[1, 0] > 0
    assert means[1, 1] > 0
    cross_products = np.cross(means[:-1], means[1:])
    assert np.all(cross_products < 0)
    np.testing.assert_allclose(np.linalg.norm(means, axis=1), 3.0)


def test_dataset_bundle_has_independent_balanced_splits() -> None:
    splits, meta = build_dataset_bundle(
        n_tasks=5,
        dim=2,
        train_per_task=20,
        val_per_task=10,
        test_per_task=15,
        data_seed=7,
    )

    for name, expected_per_task in (("train", 20), ("val", 10), ("test", 15)):
        dataset = splits[name]
        assert dataset.x.shape == (5 * expected_per_task, 2)
        np.testing.assert_array_equal(
            np.bincount(dataset.task, minlength=5),
            np.full(5, expected_per_task),
        )
    assert not np.array_equal(splits["train"].x[:10], splits["val"].x[:10])
    assert meta["task_order"] == [0, 1, 2, 3, 4]
    assert meta["ordering"] == "top_then_clockwise"


def test_equal_lambda_schedule_unrolls_to_uniform_weights() -> None:
    weights = np.asarray([1.0])
    for task_id in range(1, 5):
        weights = update_policy_weights(weights, task_id / (task_id + 1.0))
        np.testing.assert_allclose(weights, np.full(task_id + 1, 1.0 / (task_id + 1)))


def test_sequential_metrics_identify_balanced_true_samples() -> None:
    splits, _ = build_dataset_bundle(
        n_tasks=5,
        train_per_task=20,
        val_per_task=10,
        test_per_task=15,
        data_seed=7,
    )
    test = splits["test"]
    metrics = evaluate_sequential_samples(
        test.x,
        test,
        n_seen=5,
        policy_weights=np.full(5, 0.2),
    )

    assert metrics["mass_kl_uniform"] < 1e-12
    assert metrics["mass_l1_uniform"] < 1e-12
    assert categorical_kl(np.full(5, 0.2), np.full(5, 0.2)) == 0.0
