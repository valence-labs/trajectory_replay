"""Holdout replay buffer sampling (real replay support pool)."""

from __future__ import annotations

import torch

from trajectory_replay.holdout.data import (
    build_stratum_cell_index,
    build_support_flat_index,
    build_support_stratum_distribution,
    sample_support_cells,
    sample_support_conditions,
)


def _tiny_support_pool() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    x = torch.arange(12, dtype=torch.float32).unsqueeze(1)
    c = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    p = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    z = {"cell_type": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])}
    return x, c, p, z


def test_sample_support_cells_matches_conditions_proportional() -> None:
    x, c, p, z = _tiny_support_pool()
    support_contexts = [0, 1]
    stratum_dist = build_support_stratum_distribution(
        c, p, support_contexts, z=z, covariate_col_order=("cell_type",)
    )
    stratum_cells = build_stratum_cell_index(
        c, p, support_contexts, z=z, covariate_col_order=("cell_type",)
    )
    flat = build_support_flat_index(c, support_contexts)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(0)

    out = sample_support_cells(
        64,
        x,
        c,
        p,
        support_contexts=support_contexts,
        n_perturbations=2,
        device=torch.device("cpu"),
        cpu_gen=cpu_gen,
        sampling="proportional",
        stratum_dist=stratum_dist,
        stratum_cells=stratum_cells,
        support_flat_idx=flat,
        z=z,
        covariate_col_order=("cell_type",),
    )
    x_out, c_out, p_out, z_out = out
    assert x_out.shape == (64, 1)
    assert c_out.shape == (64,)
    assert p_out.shape == (64,)
    assert z_out is not None and z_out["cell_type"].shape == (64,)

    for i in range(64):
        key = (int(c_out[i]), int(p_out[i]), int(z_out["cell_type"][i]))
        assert key in stratum_cells
        row = int((x_out[i] == x).nonzero(as_tuple=True)[0].item())
        assert row in stratum_cells[key].tolist()


def test_sample_support_cells_uniform_fallback() -> None:
    x, c, p, _z = _tiny_support_pool()
    support_contexts = [0, 1]
    stratum_cells = build_stratum_cell_index(c, p, support_contexts)
    flat = build_support_flat_index(c, support_contexts)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(1)

    x_out, c_out, p_out = sample_support_cells(
        32,
        x,
        c,
        p,
        support_contexts=support_contexts,
        n_perturbations=3,
        device=torch.device("cpu"),
        cpu_gen=cpu_gen,
        sampling="uniform",
        stratum_dist=None,
        stratum_cells=stratum_cells,
        support_flat_idx=flat,
    )
    assert x_out.shape[0] == 32
    assert set(c_out.tolist()).issubset(support_contexts)
    assert all(int((x_out[i] == x).any()) for i in range(32))


def test_observed_uniform_equalizes_fat_and_thin_strata() -> None:
    """A 10-cell control pool gets the same draw rate as a 1-cell knockdown."""
    n_ctrl, n_pert = 10, 1
    x = torch.arange(n_ctrl + n_pert, dtype=torch.float32).unsqueeze(1)
    c = torch.zeros(n_ctrl + n_pert, dtype=torch.long)
    p = torch.tensor([0] * n_ctrl + [1], dtype=torch.long)
    dist = build_support_stratum_distribution(c, p, [0])
    assert dist.c.shape[0] == 2
    assert float(dist.probs[0]) > 0.8

    n = 4000
    counts = {}
    for name in ("proportional", "observed_uniform"):
        cpu_gen = torch.Generator()
        cpu_gen.manual_seed(0)
        _, p_out = sample_support_conditions(
            n,
            support_contexts=[0],
            n_perturbations=2,
            device=torch.device("cpu"),
            cpu_gen=cpu_gen,
            sampling=name,
            stratum_dist=dist,
        )
        counts[name] = int((p_out == 0).sum())

    assert counts["proportional"] > 0.7 * n
    assert abs(counts["observed_uniform"] - n / 2) < 0.08 * n


def test_stratum_sampling_probs_observed_uniform() -> None:
    from trajectory_replay.holdout.data import stratum_sampling_probs

    c = torch.tensor([0, 0, 1])
    p = torch.tensor([0, 1, 0])
    dist = build_support_stratum_distribution(c, p, [0, 1])
    # Three observed (c,p); one context/pert pair missing from the 2×2 grid.
    uni = stratum_sampling_probs(dist, "observed_uniform")
    assert uni.shape == dist.probs.shape
    assert torch.allclose(uni, torch.full_like(uni, 1.0 / uni.numel()))
    prop = stratum_sampling_probs(dist, "proportional")
    assert torch.allclose(prop, dist.probs.cpu())


def test_build_support_stratum_distribution_counts() -> None:
    x, c, p, z = _tiny_support_pool()
    del x
    dist = build_support_stratum_distribution(
        c, p, [0, 1], z=z, covariate_col_order=("cell_type",)
    )
    assert len(dist.c) == 8  # 2 contexts × 2 perturbations × 2 cell types
    assert torch.isclose(dist.probs.sum(), torch.tensor(1.0))
    assert dist.z is not None


def test_batched_sample_support_cells_same_seed_as_conditions() -> None:
    """Batched cell sampling uses one multinomial draw (same as conditions-only path)."""
    x, c, p, z = _tiny_support_pool()
    support_contexts = [0, 1]
    stratum_dist = build_support_stratum_distribution(
        c, p, support_contexts, z=z, covariate_col_order=("cell_type",)
    )
    stratum_cells = build_stratum_cell_index(
        c, p, support_contexts, z=z, covariate_col_order=("cell_type",)
    )
    flat = build_support_flat_index(c, support_contexts)

    g1 = torch.Generator()
    g1.manual_seed(42)
    c1, p1, z1 = sample_support_conditions(
        16,
        support_contexts=support_contexts,
        n_perturbations=2,
        device=torch.device("cpu"),
        cpu_gen=g1,
        sampling="proportional",
        stratum_dist=stratum_dist,
    )

    g2 = torch.Generator()
    g2.manual_seed(42)
    out = sample_support_cells(
        16,
        x,
        c,
        p,
        support_contexts=support_contexts,
        n_perturbations=2,
        device=torch.device("cpu"),
        cpu_gen=g2,
        sampling="proportional",
        stratum_dist=stratum_dist,
        stratum_cells=stratum_cells,
        support_flat_idx=flat,
        z=z,
        covariate_col_order=("cell_type",),
    )
    _, c2, p2, z2 = out
    assert torch.equal(c1, c2)
    assert torch.equal(p1, p2)
    assert z1 is not None and z2 is not None
    assert torch.equal(z1["cell_type"], z2["cell_type"])
