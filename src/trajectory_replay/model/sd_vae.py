"""Frozen Stable Diffusion VAE helpers for latent image-CL (encode/decode)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


DEFAULT_SD_VAE_ID = "stabilityai/sd-vae-ft-mse"


class FrozenSDVAE(nn.Module):
    """Thin wrapper around ``diffusers.AutoencoderKL`` used only for decode/encode.

    Latents follow the DiT / SD convention: stored & modeled as
    ``posterior.mean * scaling_factor``. Decode divides by ``scaling_factor`` first.
    Pixel tensors are in ``[-1, 1]``.
    """

    def __init__(self, vae: nn.Module, *, scaling_factor: float | None = None) -> None:
        super().__init__()
        self.vae = vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        cfg_scale = getattr(getattr(vae, "config", None), "scaling_factor", None)
        self.scaling_factor = float(
            scaling_factor if scaling_factor is not None else (cfg_scale if cfg_scale is not None else 0.18215)
        )

    @classmethod
    def from_pretrained(
        cls,
        vae_id: str = DEFAULT_SD_VAE_ID,
        *,
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> FrozenSDVAE:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(vae_id, **kwargs)
        wrapper = cls(vae)
        if device is not None:
            wrapper.to(device)
        return wrapper

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Decode scaled latents ``(N, 4, H, W)`` → RGB ``(N, 3, 8H, 8W)`` in ``[-1, 1]``."""
        if z.ndim != 4 or z.shape[1] != 4:
            raise ValueError(f"expected latents (N, 4, H, W), got {tuple(z.shape)}")
        device = next(self.vae.parameters()).device
        outs: list[torch.Tensor] = []
        n = int(z.shape[0])
        for start in range(0, n, int(batch_size)):
            zb = z[start : start + batch_size].to(device=device, dtype=torch.float32)
            zb = zb / self.scaling_factor
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                img = self.vae.decode(zb).sample
            outs.append(img.detach().float().cpu().clamp(-1.0, 1.0))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        *,
        batch_size: int = 32,
        sample_posterior: bool = False,
    ) -> torch.Tensor:
        """Encode RGB ``[-1, 1]`` → scaled latent means (or samples)."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected RGB (N, 3, H, W), got {tuple(x.shape)}")
        device = next(self.vae.parameters()).device
        outs: list[torch.Tensor] = []
        n = int(x.shape[0])
        for start in range(0, n, int(batch_size)):
            xb = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                posterior = self.vae.encode(xb).latent_dist
                z = posterior.sample() if sample_posterior else posterior.mean
                z = z * self.scaling_factor
            outs.append(z.detach().float().cpu())
        return torch.cat(outs, dim=0)


__all__ = [
    "DEFAULT_SD_VAE_ID",
    "FrozenSDVAE",
]
