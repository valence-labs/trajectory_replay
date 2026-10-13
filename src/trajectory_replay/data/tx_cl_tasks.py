"""Sequential CL task streams on Tx dumps (``obs['split']`` + context).

These streams are the paper protocol, not the old one-shot finetune:

* **Replogle (2 tasks).** T0 = jurkat / k562 / rpe1 (``split==train``).
  T1 = HepG2 few-shot genes (``split==finetune``). HepG2 STATE-test genes
  stay ``test`` forever.
* **PBMC (5 tasks).** T0 = the eight pretrain donors (``split==train``).
  T1–T4 = Donor1 → Donor4 → Donor9 → Donor12, one donor per step, using
  that donor's ``finetune`` cells only. After a step, those cells join the
  old pool. STATE-test cytokines stay ``test`` forever.

Controls (``non-targeting`` / ``PBS``) are always trainable, so every task's
pool carries its own basal. ``test`` holds no controls; they are injected at
eval time from the same context's train/finetune cells.

``test`` cells are never in a train mask. λ-grid candidates are scored on a
capped sample of the full trainable pools; no extra cell-level val split is
used.

Eval tracks after completing task ``k`` (0-based):

* **fit** — cells just trained at task ``k`` (same as ``new_train``)
* **forget** — cells trained at tasks ``0 .. k-1`` (same as ``old_train``)
* **heldout** — ``split==test`` cells whose context has already entered
  the stream (empty after T0 on the real dumps)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from trajectory_replay.data.tx_state_splits import (
    PBMC_FINETUNE_DONORS,
    REPLOGLE_FEWSHOT_CELL_LINE,
)

DatasetName = Literal["pbmc", "replogle"]

PBMC_PRETRAIN_DONORS: tuple[str, ...] = (
    "Donor2",
    "Donor3",
    "Donor5",
    "Donor6",
    "Donor7",
    "Donor8",
    "Donor10",
    "Donor11",
)
PBMC_FEWSHOT_ORDER: tuple[str, ...] = PBMC_FINETUNE_DONORS
REPLOGLE_PRETRAIN_CELL_LINES: tuple[str, ...] = ("jurkat", "k562", "rpe1")

TRAIN_SPLIT = "train"
FINETUNE_SPLIT = "finetune"
TEST_SPLIT = "test"


@dataclass(frozen=True)
class TxClTask:
    """One step of a sequential stream."""

    task_id: int
    name: str
    contexts: tuple[str, ...]
    train_split: str


@dataclass(frozen=True)
class TxClStream:
    dataset: DatasetName
    context_col: str
    perturbation_col: str
    tasks: tuple[TxClTask, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("Stream has no tasks.")
        seen: set[str] = set()
        for i, task in enumerate(self.tasks):
            if task.task_id != i:
                raise ValueError(f"Task ids must be 0..n-1 contiguous, got {task.task_id} at {i}.")
            if not task.contexts:
                raise ValueError(f"Task {i} has no contexts.")
            dup = seen.intersection(task.contexts)
            if dup:
                raise ValueError(f"Context {sorted(dup)} appears in more than one task.")
            seen.update(task.contexts)
            expected_split = TRAIN_SPLIT if i == 0 else FINETUNE_SPLIT
            if task.train_split != expected_split:
                raise ValueError(
                    f"Task {i} train_split must be {expected_split!r}, got {task.train_split!r}."
                )

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def task(self, task_id: int) -> TxClTask:
        if not 0 <= task_id < self.n_tasks:
            raise IndexError(f"task_id {task_id} out of range for {self.n_tasks} tasks.")
        return self.tasks[task_id]

    def contexts_introduced(self, task_id: int) -> tuple[str, ...]:
        return self.task(task_id).contexts

    def seen_contexts(self, through_task: int) -> tuple[str, ...]:
        ctx: list[str] = []
        for k in range(through_task + 1):
            ctx.extend(self.task(k).contexts)
        return tuple(ctx)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "context_col": self.context_col,
            "perturbation_col": self.perturbation_col,
            "tasks": [asdict(t) for t in self.tasks],
        }


@dataclass(frozen=True)
class TxClStepMasks:
    """Boolean masks for training and eval at one stream step.

    ``fit`` / ``forget`` alias ``new_train`` / ``old_train`` so later CL
    code can talk in eval-track names without remapping.
    """

    new_train: np.ndarray
    old_train: np.ndarray
    seen_train: np.ndarray
    heldout: np.ndarray

    @property
    def fit(self) -> np.ndarray:
        return self.new_train

    @property
    def forget(self) -> np.ndarray:
        return self.old_train

    def counts(self) -> dict[str, int]:
        return {
            "new_train": int(self.new_train.sum()),
            "old_train": int(self.old_train.sum()),
            "seen_train": int(self.seen_train.sum()),
            "heldout": int(self.heldout.sum()),
        }


def pbmc_stream() -> TxClStream:
    tasks = (
        TxClTask(
            task_id=0,
            name="pretrain_8donors",
            contexts=PBMC_PRETRAIN_DONORS,
            train_split=TRAIN_SPLIT,
        ),
        *(
            TxClTask(
                task_id=i + 1,
                name=donor.lower(),
                contexts=(donor,),
                train_split=FINETUNE_SPLIT,
            )
            for i, donor in enumerate(PBMC_FEWSHOT_ORDER)
        ),
    )
    return TxClStream(
        dataset="pbmc",
        context_col="donor",
        perturbation_col="cytokine",
        tasks=tasks,
    )


def replogle_stream() -> TxClStream:
    return TxClStream(
        dataset="replogle",
        context_col="cell_line",
        perturbation_col="gene",
        tasks=(
            TxClTask(
                task_id=0,
                name="pretrain_3lines",
                contexts=REPLOGLE_PRETRAIN_CELL_LINES,
                train_split=TRAIN_SPLIT,
            ),
            TxClTask(
                task_id=1,
                name=REPLOGLE_FEWSHOT_CELL_LINE,
                contexts=(REPLOGLE_FEWSHOT_CELL_LINE,),
                train_split=FINETUNE_SPLIT,
            ),
        ),
    )


def stream_for(dataset: DatasetName) -> TxClStream:
    if dataset == "pbmc":
        return pbmc_stream()
    if dataset == "replogle":
        return replogle_stream()
    raise ValueError(f"Unknown dataset {dataset!r}.")


def _as_str(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def task_train_mask(
    stream: TxClStream,
    task_id: int,
    context: np.ndarray,
    split: np.ndarray,
) -> np.ndarray:
    """Cells used as training data when learning ``task_id``."""
    task = stream.task(task_id)
    ctx = _as_str(context)
    spl = _as_str(split)
    return np.isin(ctx, task.contexts) & (spl == task.train_split)


def seen_train_mask(
    stream: TxClStream,
    through_task: int,
    context: np.ndarray,
    split: np.ndarray,
) -> np.ndarray:
    """Union of trainable cells from tasks ``0 .. through_task`` (old pool + new)."""
    mask = np.zeros(np.asarray(split).shape[0], dtype=bool)
    for k in range(through_task + 1):
        mask |= task_train_mask(stream, k, context, split)
    return mask


def heldout_mask(
    stream: TxClStream,
    through_task: int,
    context: np.ndarray,
    split: np.ndarray,
) -> np.ndarray:
    """STATE-test cells on contexts that have entered the stream by ``through_task``."""
    ctx = _as_str(context)
    spl = _as_str(split)
    return np.isin(ctx, stream.seen_contexts(through_task)) & (spl == TEST_SPLIT)


def step_masks(
    stream: TxClStream,
    task_id: int,
    context: np.ndarray,
    split: np.ndarray,
    perturbation: np.ndarray | None = None,
) -> TxClStepMasks:
    """Masks at the moment we train (and then eval) ``task_id``.

    ``perturbation`` is unused; kept so later (c,p) grouping can share this
    signature without a breaking change.
    """
    del perturbation
    new_train = task_train_mask(stream, task_id, context, split)
    if task_id == 0:
        old_train = np.zeros_like(new_train)
    else:
        old_train = seen_train_mask(stream, task_id - 1, context, split)
    seen_train = old_train | new_train
    heldout = heldout_mask(stream, task_id, context, split)
    return TxClStepMasks(
        new_train=new_train,
        old_train=old_train,
        seen_train=seen_train,
        heldout=heldout,
    )


def train_task_ids(stream: TxClStream, context: np.ndarray, split: np.ndarray) -> np.ndarray:
    """Per-cell train-task id; ``-1`` if the cell is never used for training."""
    n = np.asarray(split).shape[0]
    ids = np.full(n, -1, dtype=np.int16)
    for task in stream.tasks:
        mask = task_train_mask(stream, task.task_id, context, split)
        clash = mask & (ids >= 0)
        if np.any(clash):
            raise ValueError("A cell matched more than one train task.")
        ids[mask] = task.task_id
    return ids


def assert_test_never_trained(masks: TxClStepMasks, split: np.ndarray) -> None:
    """Invariant: ``split==test`` cells are absent from every train mask."""
    is_test = _as_str(split) == TEST_SPLIT
    for name in ("new_train", "old_train", "seen_train"):
        leaked = is_test & getattr(masks, name)
        if np.any(leaked):
            raise AssertionError(f"{name} includes {int(leaked.sum())} test cells.")


def assert_stream_test_never_trained(
    stream: TxClStream,
    context: np.ndarray,
    split: np.ndarray,
    perturbation: np.ndarray | None = None,
    *,
    n_tasks: int | None = None,
) -> None:
    """Raise if any task's train pool or ``train_task_ids`` contains STATE-test cells."""
    n = stream.n_tasks if n_tasks is None else int(n_tasks)
    for k in range(n):
        assert_test_never_trained(
            step_masks(stream, k, context, split, perturbation), split
        )
    ids = train_task_ids(stream, context, split)
    leaked = (_as_str(split) == TEST_SPLIT) & (ids >= 0)
    if np.any(leaked):
        raise AssertionError(
            f"train_task_ids assigns {int(leaked.sum())} test cells to a train task."
        )


def load_stream_obs(
    h5ad_path: Path | str,
    stream: TxClStream,
    *,
    split_col: str = "split",
) -> dict[str, np.ndarray]:
    """Read context, perturbation, and split without loading ``X``."""
    from trajectory_replay.data.tx_preprocess import (
        decode_h5ad_categorical,
    )

    path = Path(h5ad_path)
    out = {
        "context": decode_h5ad_categorical(path, stream.context_col),
        "perturbation": decode_h5ad_categorical(path, stream.perturbation_col),
        "split": decode_h5ad_categorical(path, split_col),
    }
    return out


def summarize_stream(
    stream: TxClStream,
    context: np.ndarray,
    split: np.ndarray,
    perturbation: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Cell counts per step (login-safe once obs arrays are in memory)."""
    rows: list[dict[str, Any]] = []
    for task in stream.tasks:
        masks = step_masks(stream, task.task_id, context, split, perturbation)
        assert_test_never_trained(masks, split)
        rows.append(
            {
                "task_id": task.task_id,
                "name": task.name,
                "contexts": list(task.contexts),
                "train_split": task.train_split,
                **masks.counts(),
            }
        )
    return rows
