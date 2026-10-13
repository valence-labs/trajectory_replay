"""Tests for training timing helpers."""

from __future__ import annotations

from trajectory_replay.utils.timing import training_timing_record


def test_training_timing_record() -> None:
    rec = training_timing_record(
        wall_seconds=100.0,
        epochs=10,
        steps_per_epoch=5,
        finetune_strategy="generative_replay",
        replay_backend="flow_map",
        flow_map_steps=4,
    )
    assert rec["train_wall_seconds"] == 100.0
    assert rec["train_steps"] == 50
    assert rec["seconds_per_epoch"] == 10.0
    assert rec["seconds_per_step"] == 2.0
    assert rec["replay_backend"] == "flow_map"
    assert rec["flow_map_steps"] == 4
    assert rec["mean_flow_steps"] is None

    rec_mf = training_timing_record(
        wall_seconds=50.0,
        epochs=5,
        steps_per_epoch=2,
        replay_backend="mean_flow",
        mean_flow_steps=2,
    )
    assert rec_mf["mean_flow_steps"] == 2
    assert rec_mf["flow_map_steps"] is None
