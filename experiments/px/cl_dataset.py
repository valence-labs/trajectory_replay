"""Deterministic experiment-incremental RxRx1 task construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from torch.utils.data import DataLoader, DistributedSampler

from experiments.px.dataset import CellDataset, METADATA_PATH


@dataclass(frozen=True)
class ExperimentTask:
    task_id: int
    experiment_id: int
    batch_name: str
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def _stratified_train_val(
    frame: pl.DataFrame,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    is_val = np.zeros(frame.height, dtype=bool)
    row_groups = (
        frame.with_row_index("_row")
        .group_by("perturbation_id")
        .agg(pl.col("_row"))
        .sort("perturbation_id")
    )
    for perturbation_id, indices in row_groups.iter_rows():
        indices = np.asarray(indices, dtype=np.int64)
        generator = np.random.default_rng(
            int(seed) + 1_000_003 * int(perturbation_id)
        )
        generator.shuffle(indices)
        n_val = min(
            indices.size - 1,
            max(1, int(round(indices.size * val_fraction))),
        )
        is_val[indices[:n_val]] = True
    return frame.filter(~pl.Series(is_val)), frame.filter(pl.Series(is_val))


def build_experiment_tasks(
    *,
    metadata_path: Path | str = METADATA_PATH,
    val_fraction: float = 0.1,
    split_seed: int = 0,
) -> list[ExperimentTask]:
    metadata = pl.read_csv(metadata_path)
    required = {
        "BATCH",
        "SPLIT",
        "is_eval_pert",
        "experiment_id",
        "perturbation_id",
        "path",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    tasks: list[ExperimentTask] = []
    experiment_rows = (
        metadata.select(["experiment_id", "BATCH"])
        .unique()
        .sort("experiment_id")
        .to_dicts()
    )
    for task_id, row in enumerate(experiment_rows):
        experiment_id = int(row["experiment_id"])
        batch_name = str(row["BATCH"])
        task_rows = metadata.filter(
            pl.col("experiment_id") == experiment_id
        )
        train_pool = task_rows.filter(pl.col("SPLIT") == "train")
        train, val = _stratified_train_val(
            train_pool,
            val_fraction=val_fraction,
            seed=int(split_seed) + 10_000 * experiment_id,
        )
        test = task_rows.filter(
            (pl.col("SPLIT") == "test")
            & pl.col("is_eval_pert")
            & (pl.col("ANNOT") == "treated")
        )
        if min(train.height, val.height, test.height) == 0:
            raise ValueError(
                f"Empty split for experiment {experiment_id}: "
                f"train={train.height}, val={val.height}, test={test.height}"
            )
        tasks.append(
            ExperimentTask(
                task_id=task_id,
                experiment_id=experiment_id,
                batch_name=batch_name,
                train=train,
                val=val,
                test=test,
            )
        )
    return tasks


def concatenate_task_frames(
    tasks: list[ExperimentTask],
    *,
    through_task: int,
    split: str,
) -> pl.DataFrame:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split={split!r}")
    selected = [
        getattr(task, split)
        for task in tasks
        if task.task_id <= through_task
    ]
    if not selected:
        raise ValueError(f"No tasks through task {through_task}")
    return pl.concat(selected, how="vertical")


def make_loader(
    frame: pl.DataFrame,
    *,
    train: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool | None = None,
) -> DataLoader:
    dataset = CellDataset(frame, train=train, load_control=False)
    sampler = DistributedSampler(dataset, shuffle=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=train if drop_last is None else bool(drop_last),
    )
