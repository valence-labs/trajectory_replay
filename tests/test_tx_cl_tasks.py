"""Unit tests for sequential Tx CL task streams (synthetic obs only)."""

from __future__ import annotations

import numpy as np

from trajectory_replay.data.tx_cl_tasks import (
    PBMC_FEWSHOT_ORDER,
    PBMC_PRETRAIN_DONORS,
    REPLOGLE_PRETRAIN_CELL_LINES,
    TxClStream,
    TxClTask,
    TxClStepMasks,
    assert_stream_test_never_trained,
    assert_test_never_trained,
    heldout_mask,
    pbmc_stream,
    replogle_stream,
    step_masks,
    stream_for,
    summarize_stream,
    train_task_ids,
)


def _pbmc_toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = np.array(
        [
            "Donor2",
            "Donor2",
            "Donor1",
            "Donor1",
            "Donor1",
            "Donor4",
            "Donor4",
            "Donor9",
            "Donor12",
            "Donor12",
        ],
        dtype=object,
    )
    pert = np.array(
        ["PBS", "IL-2", "PBS", "G-CSF", "C5a", "PBS", "C5a", "PBS", "PBS", "C5a"],
        dtype=object,
    )
    split = np.array(
        [
            "train",
            "train",
            "finetune",
            "finetune",
            "test",
            "finetune",
            "test",
            "finetune",
            "finetune",
            "test",
        ],
        dtype=object,
    )
    return ctx, pert, split


def _replogle_toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = np.array(
        ["k562", "jurkat", "rpe1", "hepg2", "hepg2", "hepg2"],
        dtype=object,
    )
    pert = np.array(["A", "B", "C", "D", "E", "GPS1"], dtype=object)
    split = np.array(
        ["train", "train", "train", "finetune", "finetune", "test"],
        dtype=object,
    )
    return ctx, pert, split


def test_stream_factories() -> None:
    pbmc = pbmc_stream()
    assert pbmc.n_tasks == 5
    assert pbmc.context_col == "donor"
    assert pbmc.tasks[0].contexts == PBMC_PRETRAIN_DONORS
    assert tuple(t.contexts[0] for t in pbmc.tasks[1:]) == PBMC_FEWSHOT_ORDER
    assert stream_for("replogle").n_tasks == 2
    assert replogle_stream().tasks[0].contexts == REPLOGLE_PRETRAIN_CELL_LINES


def test_rejects_overlapping_contexts() -> None:
    try:
        TxClStream(
            dataset="pbmc",
            context_col="donor",
            perturbation_col="cytokine",
            tasks=(
                TxClTask(0, "a", ("Donor2",), "train"),
                TxClTask(1, "b", ("Donor2",), "finetune"),
            ),
        )
    except ValueError as exc:
        assert "more than one task" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pbmc_step0_no_old_no_heldout() -> None:
    ctx, pert, split = _pbmc_toy()
    stream = pbmc_stream()
    masks = step_masks(stream, 0, ctx, split, pert)
    assert_test_never_trained(masks, split)
    np.testing.assert_array_equal(masks.new_train, np.array([True, True] + [False] * 8))
    assert int(masks.old_train.sum()) == 0
    assert int(masks.heldout.sum()) == 0
    assert int(masks.seen_train.sum()) == 2


def test_pbmc_donor_steps_grow_old_pool_and_heldout() -> None:
    ctx, pert, split = _pbmc_toy()
    stream = pbmc_stream()

    m1 = step_masks(stream, 1, ctx, split, pert)
    assert_test_never_trained(m1, split)
    np.testing.assert_array_equal(m1.new_train, np.array([False, False, True, True] + [False] * 6))
    assert int(m1.old_train.sum()) == 2
    assert int(m1.heldout.sum()) == 1
    assert split[m1.heldout][0] == "test" and ctx[m1.heldout][0] == "Donor1"

    m2 = step_masks(stream, 2, ctx, split, pert)
    assert_test_never_trained(m2, split)
    assert int(m2.new_train.sum()) == 1
    assert int(m2.old_train.sum()) == 4
    assert set(ctx[m2.heldout]) == {"Donor1", "Donor4"}

    m4 = step_masks(stream, 4, ctx, split, pert)
    assert_test_never_trained(m4, split)
    assert int(m4.new_train.sum()) == 1
    assert int(m4.old_train.sum()) == 6
    assert set(ctx[m4.heldout]) == {"Donor1", "Donor4", "Donor12"}
    assert int(heldout_mask(stream, 3, ctx, split).sum()) == 2


def test_replogle_two_tasks() -> None:
    ctx, pert, split = _replogle_toy()
    stream = replogle_stream()
    m0 = step_masks(stream, 0, ctx, split, pert)
    assert_test_never_trained(m0, split)
    assert int(m0.new_train.sum()) == 3
    assert int(m0.heldout.sum()) == 0

    m1 = step_masks(stream, 1, ctx, split, pert)
    assert_test_never_trained(m1, split)
    assert int(m1.new_train.sum()) == 2
    assert int(m1.old_train.sum()) == 3
    assert int(m1.heldout.sum()) == 1
    assert ctx[m1.heldout][0] == "hepg2" and pert[m1.heldout][0] == "GPS1"


def test_train_task_ids_and_fit_forget_aliases() -> None:
    ctx, pert, split = _pbmc_toy()
    stream = pbmc_stream()
    ids = train_task_ids(stream, ctx, split)
    np.testing.assert_array_equal(
        ids,
        np.array([0, 0, 1, 1, -1, 2, -1, 3, 4, -1], dtype=np.int16),
    )
    masks = step_masks(stream, 1, ctx, split, pert)
    np.testing.assert_array_equal(masks.fit, masks.new_train)
    np.testing.assert_array_equal(masks.forget, masks.old_train)


def test_test_cells_never_in_summarize() -> None:
    ctx, pert, split = _pbmc_toy()
    rows = summarize_stream(pbmc_stream(), ctx, split, pert)
    assert [r["task_id"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["heldout"] == 0
    assert rows[-1]["seen_train"] == 7
    assert_stream_test_never_trained(pbmc_stream(), ctx, split, pert)
    rctx, rpert, rsplit = _replogle_toy()
    assert_stream_test_never_trained(replogle_stream(), rctx, rsplit, rpert)


def test_assert_test_never_trained_rejects_leak() -> None:
    split = np.array(["train", "test"])
    leaked = TxClStepMasks(
        new_train=np.array([False, True]),
        old_train=np.array([False, False]),
        seen_train=np.array([False, True]),
        heldout=np.array([False, True]),
    )
    try:
        assert_test_never_trained(leaked, split)
    except AssertionError as exc:
        assert "new_train" in str(exc)
    else:
        raise AssertionError("expected AssertionError")


if __name__ == "__main__":
    test_stream_factories()
    test_rejects_overlapping_contexts()
    test_pbmc_step0_no_old_no_heldout()
    test_pbmc_donor_steps_grow_old_pool_and_heldout()
    test_replogle_two_tasks()
    test_train_task_ids_and_fit_forget_aliases()
    test_test_cells_never_in_summarize()
    test_assert_test_never_trained_rejects_leak()
    print("ok")
