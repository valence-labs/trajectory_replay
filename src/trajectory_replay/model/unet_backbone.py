"""UNet velocity backbone (adapted from torchcfm / guided-diffusion)."""

from __future__ import annotations

import torch
import torch.nn as nn

from trajectory_replay.model.unet.unet import UNetModelWrapper


class UNetVelocityBackbone(nn.Module):
    """Image UNet vector field with optional (c, p) conditioning.

    When ``use_conditioning`` is False the UNet sees only ``(x, t)``.
    Otherwise context and perturbation embeddings are projected into the
    UNet time-embedding space via ``cond_add``.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        cond_dim: int,
        *,
        num_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions: str = "14",
        dropout: float = 0.0,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(int(v) for v in image_shape)
        self.cond_dim = int(cond_dim)
        self.use_conditioning = bool(use_conditioning)
        if len(self.image_shape) != 3:
            raise ValueError(f"image_shape must be (C, H, W), got {image_shape}")
        if self.use_conditioning and self.cond_dim <= 0:
            raise ValueError("cond_dim must be positive when use_conditioning=True")
        self.num_channels = int(num_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_resolutions = attention_resolutions

        self.unet = UNetModelWrapper(
            dim=self.image_shape,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            class_cond=False,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )
        time_embed_dim = num_channels * 4
        self.cond_proj = (
            nn.Linear(cond_dim, time_embed_dim) if self.use_conditioning else None
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return velocity field with shape matching ``x`` (B, C, H, W)."""
        if x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W), got {tuple(x.shape)}")
        if t.shape != (x.shape[0],):
            raise ValueError("t must be shape (B,)")
        cond_add = None
        if self.use_conditioning:
            if cond is None:
                raise ValueError("cond is required when use_conditioning=True")
            if cond.shape != (x.shape[0], self.cond_dim):
                raise ValueError(f"cond must be (B, {self.cond_dim}), got {tuple(cond.shape)}")
            cond_add = self.cond_proj(cond)  # type: ignore[misc]
        return self.unet(t, x, cond_add=cond_add)
