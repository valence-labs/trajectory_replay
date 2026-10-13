"""Backbone builders for conditional flow matching and flow map models."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from trajectory_replay.model.unet_backbone import UNetVelocityBackbone

DEFAULT_BACKBONE_TYPE = "mlp"


class ResidualMLP(nn.Module):
    """Pre-normalized residual MLP with exactly ``depth`` linear layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError(f"residual MLP depth must be >= 3, got {depth}")
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(depth - 2)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        self.residual_scale = (depth - 2) ** -0.5

    def forward(self, x):
        h = nn.functional.silu(self.input(x))
        for block in self.blocks:
            h = h + self.residual_scale * block(h)
        return self.output(nn.functional.silu(self.output_norm(h)))


def build_mlp_sequential(
    in_dim: int,
    out_dim: int,
    *,
    hidden_dim: int,
    depth: int,
) -> nn.Sequential:
    """MLP with ``depth`` linear layers (``depth - 1`` hidden blocks + output)."""
    if depth < 2:
        raise ValueError(f"depth must be >= 2, got {depth}")
    layers: list[nn.Module] = []
    d_cur = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(d_cur, hidden_dim))
        layers.append(nn.SiLU())
        d_cur = hidden_dim
    layers.append(nn.Linear(d_cur, out_dim))
    return nn.Sequential(*layers)


def normalize_image_shape(image_shape: tuple[int, ...] | list[int]) -> tuple[int, int, int]:
    """Accept ``(H, W)`` or ``(C, H, W)``; default channel count is 1."""
    shape = tuple(int(v) for v in image_shape)
    if len(shape) == 2:
        return (1, shape[0], shape[1])
    if len(shape) == 3:
        return shape
    raise ValueError(f"image_shape must be (H, W) or (C, H, W), got {image_shape}")


def image_data_dim(image_shape: tuple[int, int, int]) -> int:
    c, h, w = image_shape
    return c * h * w


def velocity_backbone_in_dim(
    *,
    data_dim: int,
    time_emb_dim: int,
    cond_emb_dim: int,
    use_conditioning: bool = True,
    n_covariates: int = 0,
) -> int:
    dim = data_dim + time_emb_dim
    if use_conditioning:
        dim += (2 + n_covariates) * cond_emb_dim
    return dim


def flow_map_backbone_in_dim(
    *,
    data_dim: int,
    time_emb_dim: int,
    cond_emb_dim: int,
    n_covariates: int = 0,
    use_conditioning: bool = True,
) -> int:
    """Input size for dual-time models (flow map / MeanFlow)."""
    dim = data_dim + 2 * time_emb_dim
    if use_conditioning:
        dim += (2 + n_covariates) * cond_emb_dim
    return dim


def resolve_backbone_type(cfg: dict[str, Any]) -> str:
    """Return backbone type from a checkpoint config (defaults to MLP for legacy ckpts)."""
    return str(cfg.get("backbone_type", DEFAULT_BACKBONE_TYPE))


def build_velocity_backbone(
    backbone_type: str,
    *,
    in_dim: int = 0,
    out_dim: int = 0,
    hidden_dim: int = 256,
    depth: int = 4,
    image_shape: tuple[int, int, int] | None = None,
    cond_dim: int = 0,
    unet_num_channels: int = 64,
    unet_num_res_blocks: int = 2,
    unet_attention_resolutions: str = "14",
    unet_dropout: float = 0.0,
    use_conditioning: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Build a velocity-field backbone."""
    if kwargs:
        raise ValueError(f"Unexpected backbone kwargs: {sorted(kwargs)}")
    if backbone_type in (DEFAULT_BACKBONE_TYPE, "mlp"):
        return build_mlp_sequential(
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )
    if backbone_type == "residual_mlp":
        return ResidualMLP(
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )
    if backbone_type == "unet":
        if image_shape is None:
            raise ValueError("image_shape is required for unet backbone")
        return UNetVelocityBackbone(
            image_shape=image_shape,
            cond_dim=cond_dim,
            num_channels=unet_num_channels,
            num_res_blocks=unet_num_res_blocks,
            attention_resolutions=unet_attention_resolutions,
            dropout=unet_dropout,
            use_conditioning=use_conditioning,
        )
    raise ValueError(f"Unknown velocity backbone type: {backbone_type!r}")


def build_flow_map_backbone(
    backbone_type: str,
    *,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 256,
    depth: int = 4,
    **kwargs: Any,
) -> nn.Module:
    """Build a flow-map backbone. UNet flow maps are not implemented yet."""
    if backbone_type == "unet":
        raise ValueError("flow-map backbones do not support unet yet; use mlp")
    if kwargs:
        raise ValueError(f"Unexpected backbone kwargs: {sorted(kwargs)}")
    if backbone_type in (DEFAULT_BACKBONE_TYPE, "mlp"):
        return build_mlp_sequential(
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )
    if backbone_type == "residual_mlp":
        return ResidualMLP(
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )
    raise ValueError(f"Unknown flow-map backbone type: {backbone_type!r}")


def register_mlp_backbone(owner: nn.Module, module: nn.Module, attr: str = "mlp") -> None:
    """Register an ``nn.Sequential`` MLP backbone under ``attr`` (checkpoint key prefix)."""
    if not isinstance(module, nn.Sequential):
        raise TypeError(f"mlp backbone must be nn.Sequential, got {type(module).__name__}")
    setattr(owner, attr, module)


def register_backbone(owner: nn.Module, module: nn.Module, backbone_type: str) -> None:
    """Attach a backbone to a wrapper model using legacy-compatible attribute names."""
    if backbone_type in (DEFAULT_BACKBONE_TYPE, "mlp"):
        register_mlp_backbone(owner, module)
        return
    owner.backbone = module


def vector_field_module(model: nn.Module) -> nn.Module:
    """Return the trainable vector-field submodule (``mlp`` or ``backbone``)."""
    backbone_type = getattr(model, "backbone_type", DEFAULT_BACKBONE_TYPE)
    if backbone_type in (DEFAULT_BACKBONE_TYPE, "mlp"):
        return model.mlp
    return model.backbone
