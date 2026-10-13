"""Low-rank adaptation (LoRA) for the flow-matching backbones.

``nn.Linear`` layers cover the MLP vector fields used by the transcriptomics and
simulation experiments. ``nn.Conv2d`` layers cover the image UNet, where almost
all of the capacity lives in convolutions rather than in linear projections.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen base linear layer plus trainable low-rank delta: W' = W + (alpha/r) * B @ A."""

    def __init__(self, linear: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        device, dtype = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + self.scaling * lora_out

    def merge(self) -> nn.Linear:
        """Return a plain ``Linear`` with adapter weights folded into ``W`` (and ``b``)."""
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            merged = nn.Linear(
                self.linear.in_features,
                self.linear.out_features,
                bias=self.linear.bias is not None,
                device=self.linear.weight.device,
                dtype=self.linear.weight.dtype,
            )
            merged.weight.copy_(self.linear.weight + delta_w)
            if self.linear.bias is not None:
                merged.bias.copy_(self.linear.bias)
        return merged


class LoRAConv2d(nn.Module):
    """Frozen base conv plus a trainable low-rank delta: W' = W + (alpha/r) * B * A.

    ``A`` carries the base kernel geometry (stride, padding, dilation) and ``B`` is
    a 1x1 stride-1 conv, so the composition is an ordinary convolution whose kernel
    is the product of the two factors. That is what :meth:`merge` folds back.
    """

    def __init__(self, conv: nn.Conv2d, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        if conv.groups != 1:
            raise ValueError(f"LoRA supports groups=1 convolutions, got groups={conv.groups}")
        self.conv = conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        device, dtype = conv.weight.device, conv.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty(rank, conv.in_channels, *conv.kernel_size, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(conv.out_channels, rank, 1, 1, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        lora_out = F.conv2d(
            F.conv2d(
                x,
                self.lora_A,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            ),
            self.lora_B,
        )
        return out + self.scaling * lora_out

    def merge(self) -> nn.Conv2d:
        """Return a plain ``Conv2d`` with adapter weights folded into ``W``."""
        with torch.no_grad():
            a = self.lora_A.flatten(start_dim=1)
            b = self.lora_B.flatten(start_dim=1)
            delta_w = (b @ a).view_as(self.conv.weight) * self.scaling
            merged = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,  # type: ignore[arg-type]
                stride=self.conv.stride,  # type: ignore[arg-type]
                padding=self.conv.padding,  # type: ignore[arg-type]
                dilation=self.conv.dilation,  # type: ignore[arg-type]
                bias=self.conv.bias is not None,
                device=self.conv.weight.device,
                dtype=self.conv.weight.dtype,
            )
            merged.weight.copy_(self.conv.weight + delta_w)
            if self.conv.bias is not None:
                merged.bias.copy_(self.conv.bias)
        return merged


LORA_MODULES = (LoRALinear, LoRAConv2d)
LORA_TARGETS: dict[type[nn.Module], type[nn.Module]] = {
    nn.Linear: LoRALinear,
    nn.Conv2d: LoRAConv2d,
}


def iter_lora_modules(module: nn.Module) -> Iterator[nn.Module]:
    for child in module.modules():
        if isinstance(child, LORA_MODULES):
            yield child


def mlp_has_lora(module: nn.Module) -> bool:
    return any(True for _ in iter_lora_modules(module))


has_lora = mlp_has_lora


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for lor in iter_lora_modules(module):
        params.append(lor.lora_A)
        params.append(lor.lora_B)
    return params


def _target_paths(
    module: nn.Module,
    target_types: tuple[type[nn.Module], ...],
) -> list[str]:
    return [
        name
        for name, child in module.named_modules()
        if name and isinstance(child, target_types)
    ]


def _wrap_target(base: nn.Module, *, rank: int, alpha: float) -> nn.Module:
    for base_type, wrapper in LORA_TARGETS.items():
        if isinstance(base, base_type):
            return wrapper(base, rank=rank, alpha=alpha)  # type: ignore[arg-type]
    raise TypeError(f"No LoRA wrapper for {type(base).__name__}")


def _replace_child(module: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def apply_lora_to_module(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    train_output_layer: bool = False,
    target_types: tuple[type[nn.Module], ...] = (nn.Linear,),
    exclude: frozenset[str] | set[str] | None = None,
) -> nn.Module:
    """Wrap a module's target layers with LoRA, excluding its final layer by default.

    ``exclude`` names output layers explicitly, which is what the image UNet needs
    since its output convolution is not the last entry in module traversal order.
    """
    paths = _target_paths(module, target_types)
    if not paths:
        names = "/".join(t.__name__ for t in target_types)
        raise ValueError(f"Module has no {names} layers to adapt")
    skip = set(exclude or ())
    if not train_output_layer and not skip:
        skip = {paths[-1]}
    for path in paths:
        if path in skip and not train_output_layer:
            continue
        base = module.get_submodule(path)
        lor = _wrap_target(base, rank=rank, alpha=alpha)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        _replace_child(module, path, lor)
    return module


def apply_lora_to_sequential(
    mlp: nn.Sequential,
    *,
    rank: int,
    alpha: float,
    train_output_layer: bool = False,
) -> nn.Sequential:
    """Wrap hidden ``Linear`` layers in ``LoRALinear``; optionally adapt the output layer."""
    apply_lora_to_module(
        mlp,
        rank=rank,
        alpha=alpha,
        train_output_layer=train_output_layer,
    )
    return mlp


def merge_lora_module(module: nn.Module) -> nn.Module:
    """Recursively merge every LoRA wrapper back into its plain base layer."""
    paths = [
        name
        for name, child in module.named_modules()
        if name and isinstance(child, LORA_MODULES)
    ]
    for path in sorted(paths, key=lambda value: value.count("."), reverse=True):
        lor = module.get_submodule(path)
        assert isinstance(lor, LORA_MODULES)
        _replace_child(module, path, lor.merge())
    return module


def merge_lora_sequential(mlp: nn.Sequential) -> nn.Sequential:
    """Replace ``LoRALinear`` modules with merged plain ``Linear`` layers."""
    merge_lora_module(mlp)
    return mlp
