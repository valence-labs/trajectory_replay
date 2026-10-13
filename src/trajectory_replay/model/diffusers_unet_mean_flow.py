"""MeanFlow average-velocity field on Masip's Diffusers UNet2D.

Same capacity as :class:`DiffusersUNetFlowMatching`, but predicts
``u(x, r, t)`` with dual-time conditioning (MeanFlow). Diffusers exposes a
single timestep embedding; we pass ``t`` as ``timestep`` and ``r`` as
``class_labels`` with ``class_embed_type='timestep'`` so the two embeddings
are summed (same trick as image MeanFlow UNet ports).

Optional **semantic** class conditioning (ImageNet CL) uses a separate
``nn.Embedding`` summed into the same time embedding — Diffusers'
``class_labels`` slot stays reserved for ``r``.

Reference: Geng et al.; layout adapted from https://github.com/zhuyu-cs/MeanFlow
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trajectory_replay.model.diffusers_unet_fm import (
    build_masip_unet2d,
    disable_fused_sdpa_if_needed,
)


def _time_broadcast(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *([1] * (x.dim() - 1)))


class DiffusersUNetMeanFlow(nn.Module):
    """Spatial MeanFlow student duck-compatible with ``mixture_fm`` replay."""

    def __init__(
        self,
        *,
        image_size: int = 32,
        in_channels: int = 1,
        train_timesteps: int = 1000,
        unet: nn.Module | None = None,
        unet_kwargs: dict[str, Any] | None = None,
        num_classes: int = 0,
        guidance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        disable_fused_sdpa_if_needed()
        self.num_classes = int(num_classes)
        self.use_conditioning = self.num_classes > 0
        self.null_class_id = self.num_classes if self.use_conditioning else -1
        self.guidance_scale = float(guidance_scale)
        self.backbone_type = "diffusers_unet2d_mean_flow"
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.train_timesteps = int(train_timesteps)
        self.image_shape = (self.in_channels, self.image_size, self.image_size)
        self.data_dim = int(torch.tensor(self.image_shape).prod().item())

        kw = dict(unet_kwargs or {})
        kw.setdefault("sample_size", self.image_size)
        kw.setdefault("in_channels", self.in_channels)
        kw.setdefault("out_channels", self.in_channels)
        # Dual-time: r is injected via Diffusers class embedding (timestep type).
        # Do not also set num_class_embeds — that would clash with r.
        kw["class_embed_type"] = "timestep"
        kw.pop("num_class_embeds", None)
        self.unet = unet if unet is not None else build_masip_unet2d(**kw)

        # Semantic class embed (optional); dim matches UNet time embedding.
        temb_dim = int(self.unet.time_embedding.linear_2.out_features)
        if self.use_conditioning:
            self.semantic_class_embed: nn.Embedding | None = nn.Embedding(
                self.num_classes + 1, temb_dim
            )
        else:
            self.semantic_class_embed = None

    @property
    def uses_spatial_state(self) -> bool:
        return True

    def _to_unet_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Keep MeanFlow times normalized so JVP time derivatives are well scaled.

        Unlike the CFM wrapper, MeanFlow differentiates through its time
        conditioning.  Mapping ``[0, 1]`` to the DDPM ``[0, T-1]`` index range
        would multiply ``du/dr`` and ``du/dt`` by ``T-1``.  The reference
        MeanFlow implementations embed normalized times directly.
        """
        return t

    def _ensure_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 2:
            return x.view(x.shape[0], *self.image_shape)
        raise ValueError(f"x must be (B, C, H, W) or flat, got {tuple(x.shape)}")

    def _unet_forward_r_and_class(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        r_timesteps: torch.Tensor,
        class_labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """UNet2D forward with dual-time ``r`` plus optional semantic class.

        Mirrors Diffusers ``UNet2DModel.forward`` emb construction, then runs
        the same down/mid/up stack. ``r`` uses the built-in timestep class
        embed; semantic ``y`` uses ``self.semantic_class_embed``.
        """
        unet = self.unet
        # 0. center
        sample = x
        if unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time + r (+ optional semantic class)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = unet.time_proj(timesteps).to(dtype=unet.dtype)
        emb = unet.time_embedding(t_emb)

        if unet.class_embedding is None:
            raise RuntimeError("MeanFlow UNet must have class_embed_type='timestep' for r")
        r_in = r_timesteps
        if not torch.is_tensor(r_in):
            r_in = torch.tensor([r_in], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(r_in) and len(r_in.shape) == 0:
            r_in = r_in[None].to(sample.device)
        r_in = r_in * torch.ones(sample.shape[0], dtype=r_in.dtype, device=r_in.device)
        if unet.config.class_embed_type == "timestep":
            r_in = unet.time_proj(r_in)
        emb = emb + unet.class_embedding(r_in).to(dtype=unet.dtype)

        if self.semantic_class_embed is not None:
            if class_labels is None:
                raise ValueError("class labels required when MeanFlow use_conditioning=True")
            labels = class_labels.to(device=sample.device, dtype=torch.long).view(-1)
            if labels.shape[0] != sample.shape[0]:
                raise ValueError(
                    f"c must be shape (B,), got {tuple(labels.shape)} for batch {sample.shape[0]}"
                )
            emb = emb + self.semantic_class_embed(labels).to(dtype=unet.dtype)

        # 2–6. pre-process / down / mid / up / post (Diffusers UNet2DModel)
        skip_sample = sample
        sample = unet.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in unet.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb)

        skip_sample = None
        for upsample_block in unet.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)
        if skip_sample is not None:
            sample = sample + skip_sample
        return sample

    def calc_u(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Average velocity ``u(x, r, t)`` (optionally class-conditional)."""
        del p, z
        x = self._ensure_spatial(x)
        b = x.shape[0]
        if r.shape != (b,) or t.shape != (b,):
            raise ValueError(f"r and t must be shape (B,), got {tuple(r.shape)}, {tuple(t.shape)}")
        timesteps = self._to_unet_timestep(t)
        r_timesteps = self._to_unet_timestep(r)

        if not self.use_conditioning:
            return self._unet_forward_r_and_class(x, timesteps, r_timesteps, None)

        if c is None:
            raise ValueError("class labels c are required when use_conditioning=True")
        labels = c.to(device=x.device, dtype=torch.long).view(-1)
        scale = float(self.guidance_scale if guidance_scale is None else guidance_scale)
        if self.training or scale == 1.0:
            return self._unet_forward_r_and_class(x, timesteps, r_timesteps, labels)
        null = torch.full_like(labels, int(self.null_class_id))
        u_c = self._unet_forward_r_and_class(x, timesteps, r_timesteps, labels)
        u_u = self._unet_forward_r_and_class(x, timesteps, r_timesteps, null)
        return u_u + scale * (u_c - u_u)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        return self.calc_u(x, r, t, c, p, z=z, guidance_scale=guidance_scale)

    def transport(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """One MeanFlow jump: ``x_t = x_r + (t - r) * u(x_r, r, t)``."""
        x = self._ensure_spatial(x)
        dt = _time_broadcast(t - r, x)
        return x + dt * self.calc_u(x, r, t, c, p, z=z, guidance_scale=guidance_scale)

    @torch.no_grad()
    def rollout_from_x0(
        self,
        x0: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        *,
        z: dict[str, torch.Tensor] | None = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        n_steps: int = 1,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if not (0.0 <= t_start < t_end <= 1.0):
            raise ValueError(
                f"require 0 <= t_start < t_end <= 1, got t_start={t_start}, t_end={t_end}"
            )
        x = self._ensure_spatial(x0)
        device, dtype = x.device, x.dtype
        b = x.shape[0]
        if self.use_conditioning:
            if c is None:
                raise ValueError("class labels c are required when use_conditioning=True")
            if c.shape != (b,):
                raise ValueError(f"c must be shape (B,), got {tuple(c.shape)}")
        times = torch.linspace(float(t_start), float(t_end), n_steps + 1, device=device, dtype=dtype)
        for i in range(n_steps):
            r = torch.full((b,), float(times[i].item()), device=device, dtype=dtype)
            t = torch.full((b,), float(times[i + 1].item()), device=device, dtype=dtype)
            x = self.transport(x, r, t, c, p, z=z, guidance_scale=guidance_scale)
        return x

    @torch.no_grad()
    def sample_from_x0(
        self,
        x0: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        t_end: float = 1.0,
        t_start: float = 0.0,
        n_steps: int = 1,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return self.rollout_from_x0(
            x0,
            c,
            p,
            z=z,
            t_start=t_start,
            t_end=t_end,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
        )

    def sample_noise(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        c, h, w = self.image_shape
        return noise_scale * torch.randn(
            batch_size, c, h, w, device=device, dtype=dtype, generator=generator
        )

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        batch_size: int | None = None,
        n_steps: int = 1,
        noise_scale: float = 1.0,
        t_end: float = 1.0,
        generator: torch.Generator | None = None,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del p, z, kwargs
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if self.use_conditioning:
            if c is None:
                raise ValueError("class labels c are required when use_conditioning=True")
            batch_size = int(c.shape[0])
        elif batch_size is None:
            raise ValueError("batch_size required for unconditional MeanFlow sample()")
        x0 = self.sample_noise(
            int(batch_size), device=device, dtype=dtype, generator=generator, noise_scale=noise_scale
        )
        return self.sample_from_x0(
            x0, c, t_end=t_end, n_steps=n_steps, guidance_scale=guidance_scale
        )


def build_diffusers_unet_mean_flow_from_config(cfg: dict[str, Any]) -> DiffusersUNetMeanFlow:
    return DiffusersUNetMeanFlow(
        image_size=int(cfg.get("image_size", 32)),
        in_channels=int(cfg.get("in_channels", 1)),
        train_timesteps=int(cfg.get("train_timesteps", 1000)),
        unet_kwargs=cfg.get("unet_kwargs"),
        num_classes=int(cfg.get("num_classes") or 0),
        guidance_scale=float(cfg.get("guidance_scale") or 1.0),
    )


def mean_flow_unet_checkpoint_config(
    model: DiffusersUNetMeanFlow,
    *,
    parent_checkpoint: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Persist the live Diffusers config so tiny/custom UNets round-trip.
    ucfg = model.unet.config
    unet_kwargs: dict[str, Any] = {
        "layers_per_block": int(ucfg.layers_per_block),
        "block_out_channels": list(ucfg.block_out_channels),
        "norm_num_groups": int(ucfg.norm_num_groups),
        "down_block_types": list(ucfg.down_block_types),
        "up_block_types": list(ucfg.up_block_types),
        "class_embed_type": ucfg.class_embed_type,
    }
    if getattr(ucfg, "attention_head_dim", None) is not None:
        unet_kwargs["attention_head_dim"] = ucfg.attention_head_dim
    cfg: dict[str, Any] = {
        "model_type": "diffusers_unet_mean_flow",
        "parent_checkpoint": parent_checkpoint,
        "image_size": model.image_size,
        "in_channels": model.in_channels,
        "train_timesteps": model.train_timesteps,
        "image_shape": list(model.image_shape),
        "data_dim": model.data_dim,
        "backbone_type": model.backbone_type,
        "use_conditioning": bool(model.use_conditioning),
        "num_classes": int(model.num_classes),
        "guidance_scale": float(model.guidance_scale),
        "experiment": "image_cl",
        "unet_kwargs": unet_kwargs,
    }
    if extra:
        cfg.update(extra)
    return cfg


def save_diffusers_unet_mean_flow_checkpoint(
    path: Path | str,
    model: DiffusersUNetMeanFlow,
    *,
    parent_checkpoint: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": mean_flow_unet_checkpoint_config(
                model, parent_checkpoint=parent_checkpoint, extra=extra
            ),
        },
        path,
    )


def load_diffusers_unet_mean_flow_checkpoint(
    path: Path | str,
    device: torch.device,
) -> tuple[DiffusersUNetMeanFlow, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = build_diffusers_unet_mean_flow_from_config(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, cfg
