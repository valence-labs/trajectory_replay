"""MNIST processed split loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class MnistManifest:
    held_out_digit: int
    data_dim: int
    image_shape: tuple[int, int, int]
    n_contexts: int
    n_perturbations: int

    @classmethod
    def from_json(cls, path: Path) -> MnistManifest:
        with path.open() as f:
            d = json.load(f)
        raw_shape = d.get("image_shape", [28, 28])
        if len(raw_shape) == 2:
            image_shape = (1, int(raw_shape[0]), int(raw_shape[1]))
        else:
            image_shape = (int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2]))
        return cls(
            held_out_digit=int(d["held_out_digit"]),
            data_dim=int(d["data_dim"]),
            image_shape=image_shape,
            n_contexts=int(d["n_contexts"]),
            n_perturbations=int(d["n_perturbations"]),
        )


@dataclass
class MnistSplit:
    x: torch.Tensor
    c: torch.Tensor
    p: torch.Tensor


def _coerce_mnist_x(
    x: torch.Tensor,
    *,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Return float32 ``x`` with shape ``(N, C, H, W)`` (accepts flat or image tensors)."""
    x = x.float()
    c, h, w = image_shape
    data_dim = c * h * w
    if x.dim() == 2:
        if x.shape[1] != data_dim:
            raise ValueError(f"expected data_dim={data_dim}, got x.shape={tuple(x.shape)}")
        return x.view(x.shape[0], c, h, w)
    if x.dim() == 3:
        if x.shape[1:] != (h, w):
            raise ValueError(f"expected (N, {h}, {w}), got x.shape={tuple(x.shape)}")
        return x.unsqueeze(1)
    if x.dim() == 4:
        if x.shape[1:] != (c, h, w):
            raise ValueError(f"expected (N, {c}, {h}, {w}), got x.shape={tuple(x.shape)}")
        return x
    raise ValueError(f"unsupported MNIST x shape {tuple(x.shape)}")


def default_processed_dir(repo_root: Path) -> Path:
    return repo_root / "data_dir" / "mnist" / "processed"


def load_manifest(processed_dir: Path) -> MnistManifest:
    return MnistManifest.from_json(processed_dir / "manifest.json")


def load_split(processed_dir: Path, name: str) -> MnistSplit:
    if name not in ("pretrain", "finetune"):
        raise ValueError(f"name must be 'pretrain' or 'finetune', got {name!r}")
    manifest = load_manifest(processed_dir)
    payload = torch.load(processed_dir / f"{name}.pt", weights_only=False)
    x = _coerce_mnist_x(payload["x"], image_shape=manifest.image_shape)
    return MnistSplit(
        x=x,
        c=payload["c"].long(),
        p=payload["p"].long(),
    )
