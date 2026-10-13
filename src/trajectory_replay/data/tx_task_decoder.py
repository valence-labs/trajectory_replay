"""Per-task decoder ``SE → genes`` for sequential Tx CL.

The flow trains in a frozen embedding (STATE SE, SCimilarity, PCA). CellEval
needs gene space, so each CL task ``k`` has its own map ``D_k`` trained with
Adam + MSE on that task's train/finetune cells **before** the flow update.
Earlier ``D_0..D_k`` stay frozen.

``D_k`` is either an ``nn.Linear`` (:class:`TaskLinearDecoder`) or a GELU MLP
with ``depth`` hidden layers (:class:`TaskMLPDecoder`), selected by the
``hidden`` argument of :func:`fit_task_decoder`.

Task 0 is one decoder for all pretrain contexts. Each few-shot step gets its
own decoder. STATE test cells are never used to fit.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 8192
DEFAULT_EPOCHS = 500
DEFAULT_HIDDEN = 0
DEFAULT_DEPTH = 1

LINEAR_KIND = "linear"
MLP_KIND = "mlp"


def _atomic_savez(path: Path, payload: dict[str, Any]) -> None:
    """Write ``.npz`` via a same-directory temp file so a crash cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(tmp, **payload)
        tmp.replace(path)
    finally:
        if tmp.is_file():
            tmp.unlink()

BatchIter = Iterable[tuple[np.ndarray, np.ndarray]]
BatchFactory = Callable[[], BatchIter]


@dataclass(frozen=True)
class TaskLinearDecoder:
    """Frozen ``nn.Linear``: ``genes ≈ se @ weight.T + intercept``.

    ``identity=True`` is for CFM trained in gene space: decode is a no-op.
    """

    task_id: int
    weight: np.ndarray  # (n_genes, se_dim) — torch Linear layout
    intercept: np.ndarray  # (n_genes,)
    n_fit: int
    lr: float
    weight_decay: float
    epochs: int
    identity: bool = False

    @property
    def n_genes(self) -> int:
        return int(self.intercept.shape[0]) if self.identity else int(self.weight.shape[0])

    @property
    def se_dim(self) -> int:
        return int(self.intercept.shape[0]) if self.identity else int(self.weight.shape[1])

    def decode(self, se: np.ndarray) -> np.ndarray:
        z = np.asarray(se, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"se must be 2D, got shape {z.shape}")
        if z.shape[1] != self.se_dim:
            raise ValueError(f"se dim {z.shape[1]} != decoder se_dim {self.se_dim}")
        if self.identity:
            return z
        return z @ self.weight.T + self.intercept

    def to_npz_dict(self) -> dict[str, Any]:
        return {
            "kind": np.str_(LINEAR_KIND),
            "task_id": np.int64(self.task_id),
            "weight": self.weight,
            "intercept": self.intercept,
            "n_fit": np.int64(self.n_fit),
            "lr": np.float64(self.lr),
            "weight_decay": np.float64(self.weight_decay),
            "epochs": np.int64(self.epochs),
            "identity": np.bool_(self.identity),
        }

    def save(self, path: Path | str) -> None:
        _atomic_savez(Path(path), self.to_npz_dict())

    @classmethod
    def load(cls, path: Path | str) -> TaskLinearDecoder:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                task_id=int(z["task_id"]),
                weight=np.asarray(z["weight"], dtype=np.float64),
                intercept=np.asarray(z["intercept"], dtype=np.float64),
                n_fit=int(z["n_fit"]),
                lr=float(z["lr"]) if "lr" in z.files else DEFAULT_LR,
                weight_decay=float(z["weight_decay"]) if "weight_decay" in z.files else 0.0,
                epochs=int(z["epochs"]) if "epochs" in z.files else 0,
                identity=bool(z["identity"]) if "identity" in z.files else False,
            )


@dataclass(frozen=True)
class TaskMLPDecoder:
    """Frozen GELU MLP ``genes ≈ MLP(se)``, layers in torch ``Linear`` layout.

    ``weights[i] @ h + biases[i]`` with GELU after every layer but the last.
    """

    task_id: int
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    n_fit: int
    lr: float
    weight_decay: float
    epochs: int

    def __post_init__(self) -> None:
        if len(self.weights) < 2 or len(self.weights) != len(self.biases):
            raise ValueError(
                f"MLP needs >=2 layers with matching biases, got "
                f"{len(self.weights)} weights / {len(self.biases)} biases"
            )

    @property
    def identity(self) -> bool:
        return False

    @property
    def n_genes(self) -> int:
        return int(self.biases[-1].shape[0])

    @property
    def se_dim(self) -> int:
        return int(self.weights[0].shape[1])

    @property
    def hidden(self) -> int:
        return int(self.weights[0].shape[0])

    @property
    def depth(self) -> int:
        return len(self.weights) - 1

    def decode(self, se: np.ndarray) -> np.ndarray:
        z = np.asarray(se, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"se must be 2D, got shape {z.shape}")
        if z.shape[1] != self.se_dim:
            raise ValueError(f"se dim {z.shape[1]} != decoder se_dim {self.se_dim}")
        h = torch.from_numpy(np.ascontiguousarray(z))
        with torch.no_grad():
            for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
                h = F.linear(h, torch.from_numpy(w), torch.from_numpy(b))
                if i + 1 < len(self.weights):
                    h = F.gelu(h)
        return h.numpy()

    def to_npz_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": np.str_(MLP_KIND),
            "task_id": np.int64(self.task_id),
            "n_layers": np.int64(len(self.weights)),
            "n_fit": np.int64(self.n_fit),
            "lr": np.float64(self.lr),
            "weight_decay": np.float64(self.weight_decay),
            "epochs": np.int64(self.epochs),
        }
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            out[f"weight_{i}"] = w
            out[f"bias_{i}"] = b
        return out

    def save(self, path: Path | str) -> None:
        _atomic_savez(Path(path), self.to_npz_dict())

    @classmethod
    def load(cls, path: Path | str) -> TaskMLPDecoder:
        with np.load(path, allow_pickle=False) as z:
            n_layers = int(z["n_layers"])
            return cls(
                task_id=int(z["task_id"]),
                weights=tuple(
                    np.asarray(z[f"weight_{i}"], dtype=np.float64) for i in range(n_layers)
                ),
                biases=tuple(
                    np.asarray(z[f"bias_{i}"], dtype=np.float64) for i in range(n_layers)
                ),
                n_fit=int(z["n_fit"]),
                lr=float(z["lr"]),
                weight_decay=float(z["weight_decay"]),
                epochs=int(z["epochs"]),
            )


TaskDecoder = TaskLinearDecoder | TaskMLPDecoder


def load_task_decoder(path: Path | str) -> TaskDecoder:
    """Load a decoder npz of either kind (files without ``kind`` are linear)."""
    with np.load(path, allow_pickle=False) as z:
        kind = str(z["kind"]) if "kind" in z.files else LINEAR_KIND
    if kind == MLP_KIND:
        return TaskMLPDecoder.load(path)
    if kind == LINEAR_KIND:
        return TaskLinearDecoder.load(path)
    raise ValueError(f"Unknown decoder kind {kind!r} in {path}")


def identity_task_decoder(*, task_id: int, n_genes: int, n_fit: int = 0) -> TaskLinearDecoder:
    """No-op decoder when the CFM already lives in gene space."""
    g = int(n_genes)
    return TaskLinearDecoder(
        task_id=int(task_id),
        weight=np.zeros((0, 0), dtype=np.float64),
        intercept=np.zeros((g,), dtype=np.float64),
        n_fit=int(n_fit),
        lr=0.0,
        weight_decay=0.0,
        epochs=0,
        identity=True,
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _export_linear(
    layer: nn.Linear,
    *,
    task_id: int,
    n_fit: int,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> TaskLinearDecoder:
    return TaskLinearDecoder(
        task_id=int(task_id),
        weight=layer.weight.detach().cpu().numpy().astype(np.float64, copy=True),
        intercept=layer.bias.detach().cpu().numpy().astype(np.float64, copy=True),
        n_fit=int(n_fit),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
    )


def _build_head(se_dim: int, n_genes: int, hidden: int, depth: int) -> nn.Module:
    """``nn.Linear`` when ``hidden <= 0``, else ``depth`` GELU hidden layers."""
    if int(hidden) <= 0:
        return nn.Linear(int(se_dim), int(n_genes))
    if int(depth) < 1:
        raise ValueError(f"depth must be >= 1 for an MLP decoder, got {depth}")
    layers: list[nn.Module] = []
    in_dim = int(se_dim)
    for _ in range(int(depth)):
        layers.append(nn.Linear(in_dim, int(hidden)))
        layers.append(nn.GELU())
        in_dim = int(hidden)
    layers.append(nn.Linear(in_dim, int(n_genes)))
    return nn.Sequential(*layers)


def _export_head(
    head: nn.Module,
    *,
    task_id: int,
    n_fit: int,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> TaskDecoder:
    if isinstance(head, nn.Linear):
        return _export_linear(
            head,
            task_id=task_id,
            n_fit=n_fit,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
        )
    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    return TaskMLPDecoder(
        task_id=int(task_id),
        weights=tuple(
            m.weight.detach().cpu().numpy().astype(np.float64, copy=True) for m in linears
        ),
        biases=tuple(m.bias.detach().cpu().numpy().astype(np.float64, copy=True) for m in linears),
        n_fit=int(n_fit),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
    )


def _train_one_epoch(
    layer: nn.Module,
    opt: torch.optim.Optimizer,
    batches: BatchIter,
    device: torch.device,
) -> int:
    n_seen = 0
    layer.train()
    for se, genes in batches:
        z = torch.as_tensor(np.asarray(se), dtype=torch.float32, device=device)
        x = torch.as_tensor(np.asarray(genes), dtype=torch.float32, device=device)
        if z.ndim != 2 or x.ndim != 2:
            raise ValueError("se and genes batches must be 2D.")
        if z.shape[0] != x.shape[0]:
            raise ValueError(f"n_cells mismatch: se {z.shape[0]} vs genes {x.shape[0]}")
        if z.shape[0] == 0:
            continue
        loss = F.mse_loss(layer(z), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n_seen += int(z.shape[0])
    return n_seen


def fit_task_decoder(
    se: np.ndarray,
    genes: np.ndarray,
    *,
    task_id: int,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    hidden: int = DEFAULT_HIDDEN,
    depth: int = DEFAULT_DEPTH,
    device: str = "cpu",
    seed: int = 0,
) -> TaskDecoder:
    """Train ``se_dim → n_genes`` with Adam + MSE on in-memory rows.

    ``hidden <= 0`` gives a plain ``nn.Linear``; otherwise a GELU MLP with
    ``depth`` hidden layers of width ``hidden``.
    """
    z = np.asarray(se)
    x = np.asarray(genes)
    if z.ndim != 2 or x.ndim != 2:
        raise ValueError("se and genes must be 2D.")
    if z.shape[0] != x.shape[0]:
        raise ValueError(f"n_cells mismatch: se {z.shape[0]} vs genes {x.shape[0]}")
    if z.shape[0] == 0:
        raise ValueError("No cells to fit the decoder.")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    g = torch.Generator()
    g.manual_seed(int(seed))
    n = int(z.shape[0])
    bs = min(int(batch_size), n)

    def epoch_batches() -> BatchIter:
        perm = torch.randperm(n, generator=g).numpy()
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            yield z[idx], x[idx]

    return fit_task_decoder_from_batches(
        epoch_batches,
        task_id=task_id,
        se_dim=int(z.shape[1]),
        n_genes=int(x.shape[1]),
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        hidden=hidden,
        depth=depth,
        device=device,
    )


def fit_task_decoder_from_batches(
    batches: BatchIter | BatchFactory,
    *,
    task_id: int,
    se_dim: int | None = None,
    n_genes: int | None = None,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    epochs: int = DEFAULT_EPOCHS,
    hidden: int = DEFAULT_HIDDEN,
    depth: int = DEFAULT_DEPTH,
    device: str = "cpu",
) -> TaskDecoder:
    """Train the decoder from minibatches (RAM = one batch + the head).

    Pass a **factory** ``() -> iterator`` so each epoch can re-read from disk.
    A list/tuple is replayed in-memory each epoch.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    dev = _device(device)
    layer: nn.Module | None = None
    opt: torch.optim.Optimizer | None = None
    cached: list[tuple[np.ndarray, np.ndarray]] | None = None
    n_fit = 0

    def epoch_batches() -> BatchIter:
        nonlocal cached
        if callable(batches) and not isinstance(batches, (list, tuple)):
            return batches()
        if isinstance(batches, (list, tuple)):
            return batches
        if cached is None:
            cached = list(batches)
        return cached

    for _ in range(int(epochs)):
        n_fit = 0
        for se_b, x_b in epoch_batches():
            z0 = np.asarray(se_b)
            x0 = np.asarray(x_b)
            if layer is None:
                d = int(se_dim) if se_dim is not None else int(z0.shape[1])
                gdim = int(n_genes) if n_genes is not None else int(x0.shape[1])
                layer = _build_head(d, gdim, hidden, depth).to(dev)
                opt = torch.optim.Adam(layer.parameters(), lr=lr, weight_decay=weight_decay)
            assert opt is not None
            n_fit += _train_one_epoch(layer, opt, [(z0, x0)], dev)
        if layer is None:
            raise ValueError("No cells to fit the decoder.")

    assert layer is not None
    return _export_head(
        layer,
        task_id=task_id,
        n_fit=n_fit,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
    )


class TaskDecoderBank:
    """Frozen stack of per-task decoders; only ``add`` for the next task id."""

    def __init__(self) -> None:
        self._decoders: dict[int, TaskDecoder] = {}

    def __len__(self) -> int:
        return len(self._decoders)

    def __contains__(self, task_id: int) -> bool:
        return int(task_id) in self._decoders

    def get(self, task_id: int) -> TaskDecoder:
        tid = int(task_id)
        if tid not in self._decoders:
            raise KeyError(f"No decoder for task {tid}.")
        return self._decoders[tid]

    def add(self, decoder: TaskDecoder) -> None:
        tid = int(decoder.task_id)
        if tid in self._decoders:
            raise ValueError(f"Decoder for task {tid} is frozen.")
        expected = 0 if not self._decoders else max(self._decoders) + 1
        if tid != expected:
            raise ValueError(f"Next decoder must be task {expected}, got {tid}.")
        self._decoders[tid] = decoder

    def decode(self, se: np.ndarray, task_ids: np.ndarray) -> np.ndarray:
        """Decode rows with the decoder of each row's train-task id."""
        z = np.asarray(se, dtype=np.float64)
        ids = np.asarray(task_ids)
        if z.shape[0] != ids.shape[0]:
            raise ValueError("se and task_ids length mismatch.")
        if z.shape[0] == 0:
            n_genes = self.get(0).n_genes if self._decoders else 0
            return np.zeros((0, n_genes), dtype=np.float64)
        n_genes = next(iter(self._decoders.values())).n_genes
        out = np.empty((z.shape[0], n_genes), dtype=np.float64)
        for tid in np.unique(ids):
            if int(tid) < 0:
                raise ValueError("Cannot decode cells with train-task id -1 (never trained).")
            mask = ids == tid
            out[mask] = self.get(int(tid)).decode(z[mask])
        return out

    def save_dir(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for tid, dec in self._decoders.items():
            dest = directory / f"decoder_task{tid:02d}.npz"
            if dest.is_file():
                continue
            dec.save(dest)

    @classmethod
    def load_dir(cls, directory: Path | str) -> TaskDecoderBank:
        directory = Path(directory)
        bank = cls()
        paths = sorted(directory.glob("decoder_task*.npz"))
        if not paths:
            raise FileNotFoundError(f"No decoder_task*.npz under {directory}")
        for path in paths:
            bank.add(load_task_decoder(path))
        return bank
