"""Image FID helpers for generative models (Masip-comparable)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

# Maps model/state tensors to RGB pixels in [-1, 1] for Inception FID.
# For latent FM this is VAE decode; for pixel FM leave as None (identity).
ToPixelsFn = Callable[[torch.Tensor], torch.Tensor]


def _to_01_rgb(x: torch.Tensor) -> torch.Tensor:
    """Map model images in [-1, 1] or [0, 1] (C=1 or 3) to RGB in [0, 1]."""
    x = x.detach()
    if x.min() < 0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def _maybe_to_pixels(x: torch.Tensor, to_pixels: ToPixelsFn | None) -> torch.Tensor:
    if to_pixels is None:
        return x
    return to_pixels(x)


def sample_labels_like(
    y_ref: torch.Tensor,
    n: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``n`` labels with replacement from the empirical distribution of ``y_ref``."""
    y_ref = y_ref.detach().long().view(-1)
    if y_ref.numel() == 0:
        raise ValueError("y_ref must be non-empty")
    idx = torch.randint(0, int(y_ref.numel()), (int(n),), generator=generator)
    return y_ref[idx].clone()


def sample_labels_balanced(
    class_ids: Sequence[int],
    n: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cycle through ``class_ids`` to build ``n`` labels (near-uniform coverage)."""
    ids = [int(c) for c in class_ids]
    if not ids:
        raise ValueError("class_ids must be non-empty")
    if n <= 0:
        return torch.zeros(0, dtype=torch.long)
    # Repeat the class list enough times, then trim; shuffle for variety.
    reps = (int(n) + len(ids) - 1) // len(ids)
    labels = torch.tensor(ids * reps, dtype=torch.long)[: int(n)]
    perm = torch.randperm(int(n), generator=generator)
    return labels[perm]


@torch.no_grad()
def compute_fid(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    """Frechet Inception Distance via torchmetrics (feature=2048, normalize=True)."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(normalize=True, feature=2048).to(device)
    # Update in chunks to limit peak memory.
    chunk = 256
    for start in range(0, real.shape[0], chunk):
        fid.update(_to_01_rgb(real[start : start + chunk].to(device)), real=True)
    for start in range(0, fake.shape[0], chunk):
        fid.update(_to_01_rgb(fake[start : start + chunk].to(device)), real=False)
    return float(fid.compute().detach().cpu().item())


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    n: int,
    *,
    device: torch.device,
    batch_size: int = 128,
    ode_method: str = "euler",
    ode_step_size: float = 0.1,
    mean_flow_n_steps: int | None = None,
    seed: int = 0,
    to_pixels: ToPixelsFn | None = None,
    class_labels: torch.Tensor | None = None,
    label_source: torch.Tensor | None = None,
    class_ids: Sequence[int] | None = None,
    guidance_scale: float | None = None,
) -> torch.Tensor:
    """Generate ``n`` samples in ``[-1, 1]`` (decoded RGB if ``to_pixels`` is set).

    For class-conditional models, provide one of:
    - ``class_labels`` — exact length-``n`` label vector, or
    - ``label_source`` — sample labels i.i.d. from this empirical distribution, or
    - ``class_ids`` — near-uniform balanced labels over these class ids.
    """
    model.eval()
    use_cond = bool(getattr(model, "use_conditioning", False))
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(seed)
    labels_all: torch.Tensor | None = None
    if use_cond:
        if class_labels is not None:
            labels_all = class_labels.detach().long().view(-1)
            if int(labels_all.numel()) != int(n):
                raise ValueError(
                    f"class_labels length {int(labels_all.numel())} != n={int(n)}"
                )
        elif label_source is not None:
            labels_all = sample_labels_like(label_source, int(n), generator=cpu_gen)
        elif class_ids is not None:
            labels_all = sample_labels_balanced(class_ids, int(n), generator=cpu_gen)
        else:
            raise ValueError(
                "class-conditional generate_samples requires class_labels, "
                "label_source, or class_ids"
            )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    # Fixed-step solvers need step_size; adaptive ones (dopri5, …) must not get it.
    opts: dict[str, float] | None = None
    if str(ode_method) in ("euler", "midpoint", "rk4"):
        opts = {"step_size": float(ode_step_size)}
    chunks: list[torch.Tensor] = []
    remaining = int(n)
    offset = 0
    while remaining > 0:
        b = min(batch_size, remaining)
        sample_kwargs: dict[str, Any] = {
            "batch_size": b,
            "generator": gen,
        }
        if mean_flow_n_steps is None:
            sample_kwargs.update(method=ode_method, options=opts)
        else:
            sample_kwargs["n_steps"] = int(mean_flow_n_steps)
        if labels_all is not None:
            sample_kwargs["c"] = labels_all[offset : offset + b].to(device)
        if guidance_scale is not None:
            sample_kwargs["guidance_scale"] = float(guidance_scale)
        x = model.sample(**sample_kwargs)
        x = _maybe_to_pixels(x.detach(), to_pixels)
        chunks.append(x.detach().cpu())
        remaining -= b
        offset += b
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def fid_against_reals(
    model: nn.Module,
    real: torch.Tensor,
    *,
    device: torch.device,
    n_gen: int | None = None,
    batch_size: int = 128,
    ode_method: str = "euler",
    ode_step_size: float = 0.1,
    mean_flow_n_steps: int | None = None,
    seed: int = 0,
    to_pixels: ToPixelsFn | None = None,
    label_source: torch.Tensor | None = None,
    class_ids: Sequence[int] | None = None,
    guidance_scale: float | None = None,
) -> float:
    """FID between ``real`` images and ``n_gen`` model samples.

    Real and fake counts need not match (torchmetrics FID uses separate feature
    statistics). ``n_gen`` defaults to ``len(real)``.

    For latent FM, pass ``to_pixels`` so both real latents and generated latents
    are decoded to RGB before Inception features are computed.
    """
    n = int(n_gen if n_gen is not None else real.shape[0])
    fake = generate_samples(
        model,
        n,
        device=device,
        batch_size=batch_size,
        ode_method=ode_method,
        ode_step_size=ode_step_size,
        mean_flow_n_steps=mean_flow_n_steps,
        seed=seed,
        to_pixels=to_pixels,
        label_source=label_source,
        class_ids=class_ids,
        guidance_scale=guidance_scale,
    )
    real_pix = _maybe_to_pixels(real.detach(), to_pixels)
    return compute_fid(real_pix, fake, device=device)


@torch.no_grad()
def macro_average_task_fid(
    model: nn.Module,
    task_reals: list[torch.Tensor],
    *,
    device: torch.device,
    n_gen_per_task: int,
    batch_size: int = 128,
    ode_method: str = "euler",
    ode_step_size: float = 0.1,
    mean_flow_n_steps: int | None = None,
    seed: int = 0,
    to_pixels: ToPixelsFn | None = None,
    task_label_sources: list[torch.Tensor] | None = None,
    task_class_ids: list[Sequence[int]] | None = None,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    """Mean FID across tasks (equal weight). Also returns per-task FIDs."""
    per_task: list[float] = []
    for i, real in enumerate(task_reals):
        label_source = None
        class_ids = None
        if task_label_sources is not None:
            label_source = task_label_sources[i]
        elif task_class_ids is not None:
            class_ids = task_class_ids[i]
        fid_i = fid_against_reals(
            model,
            real,
            device=device,
            n_gen=n_gen_per_task,
            batch_size=batch_size,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
            mean_flow_n_steps=mean_flow_n_steps,
            seed=seed + 17 * i,
            to_pixels=to_pixels,
            label_source=label_source,
            class_ids=class_ids,
            guidance_scale=guidance_scale,
        )
        per_task.append(fid_i)
    return {
        "fid_macro": float(sum(per_task) / max(len(per_task), 1)),
        "fid_per_task": per_task,
    }


@torch.no_grad()
def mean_per_class_fid(
    model: nn.Module,
    x_real: torch.Tensor,
    y_real: torch.Tensor,
    *,
    device: torch.device,
    class_ids: Sequence[int],
    n_gen_per_class: int,
    batch_size: int = 128,
    ode_method: str = "euler",
    ode_step_size: float = 0.1,
    mean_flow_n_steps: int | None = None,
    seed: int = 0,
    to_pixels: ToPixelsFn | None = None,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    """Average FID over classes: for each ``y``, FID(reals_y, gens_y).

    Class-conditional models generate with label ``y``. Unconditional models
    generate without labels (same protocol, but gens are not class-specific).
    Classes with fewer than 2 real images are skipped.
    """
    y_real = y_real.detach().long().view(-1)
    if x_real.shape[0] != y_real.shape[0]:
        raise ValueError(
            f"x_real/y_real length mismatch: {x_real.shape[0]} vs {y_real.shape[0]}"
        )
    use_cond = bool(getattr(model, "use_conditioning", False))
    per_class: dict[str, float] = {}
    skipped: list[int] = []
    for i, cid in enumerate(class_ids):
        cid = int(cid)
        mask = y_real == cid
        n_real = int(mask.sum().item())
        if n_real < 2:
            skipped.append(cid)
            continue
        real_c = x_real[mask]
        labels = (
            torch.full((int(n_gen_per_class),), cid, dtype=torch.long)
            if use_cond
            else None
        )
        fake = generate_samples(
            model,
            int(n_gen_per_class),
            device=device,
            batch_size=batch_size,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
            mean_flow_n_steps=mean_flow_n_steps,
            seed=seed + 31 * i + cid,
            to_pixels=to_pixels,
            class_labels=labels,
            guidance_scale=guidance_scale if use_cond else None,
        )
        real_pix = _maybe_to_pixels(real_c.detach(), to_pixels)
        per_class[str(cid)] = compute_fid(real_pix, fake, device=device)
    vals = list(per_class.values())
    return {
        "fid_per_class_mean": float(sum(vals) / max(len(vals), 1)) if vals else float("nan"),
        "fid_per_class": per_class,
        "n_gen_per_class": int(n_gen_per_class),
        "n_classes_evaluated": len(vals),
        "classes_skipped": skipped,
    }
