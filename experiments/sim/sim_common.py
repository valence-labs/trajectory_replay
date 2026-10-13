"""Shared simulation training helpers."""

from __future__ import annotations

import numpy as np

from trajectory_replay.data.simulate import (
    MeanCoupling,
    make_additive_coupling,
    make_hadamard_interaction_coupling,
    make_rotated_additive_coupling,
)


def build_coupling(
    rng: np.random.Generator,
    name: str,
    *,
    n_contexts: int,
    n_perturbations: int,
    dim: int,
    context_scale: float,
    pert_scale: float,
    interaction_scale: float,
) -> MeanCoupling:
    if name == "additive":
        return make_additive_coupling(
            rng,
            n_contexts=n_contexts,
            n_perturbations=n_perturbations,
            dim=dim,
            context_scale=context_scale,
            pert_scale=pert_scale,
        )
    if name == "rotated":
        return make_rotated_additive_coupling(
            rng,
            n_contexts=n_contexts,
            n_perturbations=n_perturbations,
            dim=dim,
            context_scale=context_scale,
            pert_scale=pert_scale,
        )
    if name == "hadamard":
        return make_hadamard_interaction_coupling(
            rng,
            n_contexts=n_contexts,
            n_perturbations=n_perturbations,
            dim=dim,
            context_scale=context_scale,
            pert_scale=pert_scale,
            interaction_scale=interaction_scale,
        )
    raise ValueError(f"Unknown coupling: {name}")
