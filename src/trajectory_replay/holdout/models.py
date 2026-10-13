"""Model wrappers for evaluating / plotting pretrain checkpoints with full (c, p) indices."""

from __future__ import annotations

import torch
import torch.nn as nn

from trajectory_replay.holdout.data import remap_context_indices, remap_perturbation_indices
from trajectory_replay.model.flow_matching import ConditionalFlowMatching


class RemappedPerturbationModel(nn.Module):
    """Wrap a pretrain model (n_p-1 embeddings) for plotting with full perturbation indices."""

    def __init__(self, inner: ConditionalFlowMatching, held_out: int) -> None:
        super().__init__()
        self.inner = inner
        self.held_out = held_out

    def eval(self) -> RemappedPerturbationModel:
        self.inner.eval()
        return self

    def train(self, mode: bool = True) -> RemappedPerturbationModel:
        self.inner.train(mode)
        return self

    @torch.no_grad()
    def sample(self, c: torch.Tensor, p: torch.Tensor, **kwargs: object) -> torch.Tensor:
        if (p == self.held_out).any():
            raise ValueError("cannot sample held-out perturbation with pretrain model")
        p_m = remap_perturbation_indices(p, self.held_out)
        return self.inner.sample(c, p_m, **kwargs)  # type: ignore[arg-type]


class RemappedContextModel(nn.Module):
    """Wrap a pretrain model (n_c-1 context embeddings) for plotting with full context indices."""

    def __init__(self, inner: ConditionalFlowMatching, held_out: int) -> None:
        super().__init__()
        self.inner = inner
        self.held_out = held_out

    def eval(self) -> RemappedContextModel:
        self.inner.eval()
        return self

    def train(self, mode: bool = True) -> RemappedContextModel:
        self.inner.train(mode)
        return self

    @torch.no_grad()
    def sample(self, c: torch.Tensor, p: torch.Tensor, **kwargs: object) -> torch.Tensor:
        if (c == self.held_out).any():
            raise ValueError("cannot sample held-out context with pretrain model")
        c_m = remap_context_indices(c, self.held_out)
        return self.inner.sample(c_m, p, **kwargs)  # type: ignore[arg-type]
