"""CFM velocity field wrapping Masip's ``diffusers.UNet2DModel``.

Matches ``configs/model/ddim_medium.json`` capacity so FM image-CL experiments
are backbone-comparable to the diffusion baselines. Continuous CFM time
``t ∈ [0, 1]`` is mapped to the UNet timestep embedding via ``t * (train_timesteps - 1)``.

Optional class conditioning uses Diffusers ``num_class_embeds`` (semantic labels),
with an extra null class for classifier-free guidance. This is distinct from
MeanFlow's reuse of ``class_labels`` for the second time coordinate.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import torch
import torch.nn as nn
from torchdiffeq import odeint

from trajectory_replay.model.lora import (
    apply_lora_to_module,
    has_lora,
    merge_lora_module,
)

# Masip ddim_medium.json defaults.
DEFAULT_DDIM_MEDIUM: dict[str, Any] = {
    "sample_size": 32,
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": 1,
    "block_out_channels": (64, 128, 256, 512),
    "norm_num_groups": 32,
    "down_block_types": (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
    ),
    "up_block_types": (
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
}

# Masip configs/model/ddim_medium_3ch.json defaults for CIFAR-10.
DEFAULT_DDIM_MEDIUM_3CH: dict[str, Any] = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "block_out_channels": (128, 256, 256, 256),
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "freq_shift": 1,
    "attention_head_dim": None,
    "flip_sin_to_cos": False,
    "down_block_types": (
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
}

# ImageNet-100 latent FM: Masip 4ch width/depth/block layout, but Lipman-style
# multi-head attention (head dim 64) instead of Masip's single-head
# ``attention_head_dim=None``. CIFAR/FMNIST keep Masip's single-head setting.
DEFAULT_DDIM_MEDIUM_4CH: dict[str, Any] = {
    **DEFAULT_DDIM_MEDIUM_3CH,
    "in_channels": 4,
    "out_channels": 4,
    "attention_head_dim": 64,
}

# Larger latent UNet for ImageNet-100 (~3× medium params). More width + attention
# on the two lowest resolutions (LDM-style), still 32×32×4.
DEFAULT_DDIM_LARGE_4CH: dict[str, Any] = {
    "sample_size": 32,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 2,
    "block_out_channels": (128, 256, 512, 512),
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "freq_shift": 1,
    "attention_head_dim": 64,
    "flip_sin_to_cos": False,
    "down_block_types": (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    "up_block_types": (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
}


def disable_fused_sdpa_if_needed() -> None:
    """torch 2.0.1 fused SDPA miscompiles on sm_90 (H100); see Masip baseline patch."""
    if os.environ.get("DISABLE_FUSED_SDPA", "1") != "1":
        return
    if not hasattr(torch.backends.cuda, "enable_flash_sdp"):
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def build_masip_unet2d(**overrides: Any) -> nn.Module:
    """Construct HuggingFace ``UNet2DModel`` with Masip medium defaults.

    Extra Diffusers kwargs (e.g. ``class_embed_type='timestep'`` for MeanFlow
    dual-time) are forwarded via ``overrides``.
    """
    from diffusers import UNet2DModel

    cfg = {**DEFAULT_DDIM_MEDIUM, **overrides}
    return UNet2DModel(
        sample_size=int(cfg["sample_size"]),
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        layers_per_block=int(cfg["layers_per_block"]),
        block_out_channels=tuple(cfg["block_out_channels"]),
        norm_num_groups=int(cfg["norm_num_groups"]),
        down_block_types=tuple(cfg["down_block_types"]),
        up_block_types=tuple(cfg["up_block_types"]),
        **{
            k: v
            for k, v in overrides.items()
            if k
            not in {
                "sample_size",
                "in_channels",
                "out_channels",
                "layers_per_block",
                "block_out_channels",
                "norm_num_groups",
                "down_block_types",
                "up_block_types",
            }
        },
    )


class DiffusersUNetFlowMatching(nn.Module):
    """Image CFM model duck-compatible with ``mixture_fm`` trainers.

    Exposes the same sampling / noise / forward surface as
    :class:`~trajectory_replay.model.flow_matching.ConditionalFlowMatching`.
    When ``num_classes > 0``, ``c`` is a batch of integer class labels passed to
    Diffusers as ``class_labels``.
    """

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
        # Null-class index for CFG (appended after the real classes).
        self.null_class_id = self.num_classes if self.use_conditioning else -1
        self.guidance_scale = float(guidance_scale)
        self.backbone_type = "diffusers_unet2d"
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.train_timesteps = int(train_timesteps)
        self.image_shape = (self.in_channels, self.image_size, self.image_size)
        self.data_dim = int(math.prod(self.image_shape))
        kw = dict(unet_kwargs or {})
        kw.setdefault("sample_size", self.image_size)
        kw.setdefault("in_channels", self.in_channels)
        kw.setdefault("out_channels", self.in_channels)
        if self.use_conditioning:
            # +1 embed for the CFG null token.
            kw["num_class_embeds"] = self.num_classes + 1
        self.unet = unet if unet is not None else build_masip_unet2d(**kw)

    @property
    def uses_spatial_state(self) -> bool:
        return True

    def _to_unet_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Map CFM time in [0, 1] to the UNet timestep embedding scale."""
        return t * float(self.train_timesteps - 1)

    def _unet_forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None,
    ) -> torch.Tensor:
        if class_labels is None:
            return self.unet(x, timesteps, return_dict=False)[0]
        return self.unet(x, timesteps, class_labels=class_labels, return_dict=False)[0]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        del p, z
        if t.shape != (x.shape[0],):
            raise ValueError(f"t must be shape (B,), got {tuple(t.shape)}")
        if x.dim() == 2:
            x = x.view(x.shape[0], *self.image_shape)
        elif x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W) or flat, got {tuple(x.shape)}")
        timesteps = self._to_unet_timestep(t)

        if not self.use_conditioning:
            return self._unet_forward(x, timesteps, None)

        if c is None:
            raise ValueError("class labels c are required when use_conditioning=True")
        labels = c.to(device=x.device, dtype=torch.long).view(-1)
        if labels.shape[0] != x.shape[0]:
            raise ValueError(
                f"c must be shape (B,), got {tuple(labels.shape)} for batch {x.shape[0]}"
            )

        scale = float(self.guidance_scale if guidance_scale is None else guidance_scale)
        # During training (or CFG scale≈1) run a single conditional forward.
        if self.training or scale == 1.0:
            return self._unet_forward(x, timesteps, labels)

        # Classifier-free guidance at sample time.
        null = torch.full_like(labels, int(self.null_class_id))
        v_c = self._unet_forward(x, timesteps, labels)
        v_u = self._unet_forward(x, timesteps, null)
        return v_u + scale * (v_c - v_u)

    def apply_unet_lora(
        self,
        *,
        rank: int,
        alpha: float,
        train_output_layer: bool = False,
    ) -> None:
        """Wrap UNet convolutions and linears with LoRA (base weights stay frozen).

        The output convolution is excluded unless ``train_output_layer`` is set,
        mirroring the MLP adapters used by the transcriptomics experiments.
        """
        if has_lora(self.unet):
            raise RuntimeError("UNet already has LoRA adapters")
        # Freeze everything first so normalization layers and the output conv do
        # not silently stay trainable alongside the adapters.
        for parameter in self.unet.parameters():
            parameter.requires_grad_(False)
        apply_lora_to_module(
            self.unet,
            rank=rank,
            alpha=alpha,
            train_output_layer=train_output_layer,
            target_types=(nn.Linear, nn.Conv2d),
            exclude={"conv_out"},
        )

    def merge_unet_lora(self) -> None:
        """Fold LoRA deltas into the base weights and drop the adapter modules."""
        merge_lora_module(self.unet)
        for parameter in self.unet.parameters():
            parameter.requires_grad_(True)

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
    def sample_from_x0(
        self,
        x0: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        t_end: float = 1.0,
        method: str = "euler",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        options: dict[str, Any] | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        del p, z
        if not (0.0 < t_end <= 1.0):
            raise ValueError(f"t_end must be in (0, 1], got {t_end}")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if x0.dim() == 2:
            x_init = x0.to(device=device, dtype=dtype).view(x0.shape[0], *self.image_shape)
        else:
            x_init = x0.to(device=device, dtype=dtype)
        labels = None
        if self.use_conditioning:
            if c is None:
                raise ValueError("class labels c are required for class-conditional sample_from_x0")
            labels = c.to(device=device, dtype=torch.long).view(-1)
            if labels.shape[0] != x_init.shape[0]:
                raise ValueError(
                    f"c batch {labels.shape[0]} != x0 batch {x_init.shape[0]}"
                )
        t_span = torch.tensor([0.0, float(t_end)], device=device, dtype=dtype)
        scale = guidance_scale

        def ode_func(t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            tb = torch.full((x.shape[0],), float(t_scalar.item()), device=x.device, dtype=dtype)
            return self.forward(x, tb, labels, guidance_scale=scale)

        sol = odeint(
            ode_func,
            x_init,
            t_span,
            method=method,
            rtol=rtol,
            atol=atol,
            options=options,
        )
        return sol[-1]

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
        *,
        batch_size: int | None = None,
        noise_scale: float = 1.0,
        t_end: float = 1.0,
        method: str = "euler",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        options: dict[str, Any] | None = None,
        generator: torch.Generator | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        del p, z
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if batch_size is not None:
            b = int(batch_size)
        elif c is not None:
            b = int(c.shape[0])
        else:
            raise ValueError("batch_size or c must be provided for sample()")
        if self.use_conditioning and c is None:
            raise ValueError("class labels c are required for class-conditional sample()")
        if c is not None and int(c.shape[0]) != b:
            raise ValueError(f"c batch {int(c.shape[0])} != batch_size {b}")
        x0 = self.sample_noise(
            b, device=device, dtype=dtype, generator=generator, noise_scale=noise_scale
        )
        return self.sample_from_x0(
            x0,
            c,
            t_end=t_end,
            method=method,
            rtol=rtol,
            atol=atol,
            options=options,
            guidance_scale=guidance_scale,
        )


def make_frozen_diffusers_copy(model: DiffusersUNetFlowMatching) -> DiffusersUNetFlowMatching:
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def build_diffusers_unet_fm_from_config(cfg: dict[str, Any]) -> DiffusersUNetFlowMatching:
    num_classes = cfg.get("num_classes", 0)
    if num_classes is None:
        num_classes = 0
    guidance_scale = cfg.get("guidance_scale", 1.0)
    if guidance_scale is None:
        guidance_scale = 1.0
    return DiffusersUNetFlowMatching(
        image_size=int(cfg.get("image_size", 32)),
        in_channels=int(cfg.get("in_channels", 1)),
        train_timesteps=int(cfg.get("train_timesteps", 1000)),
        unet_kwargs=cfg.get("unet_kwargs"),
        num_classes=int(num_classes),
        guidance_scale=float(guidance_scale),
    )
