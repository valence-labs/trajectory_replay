"""Conditional flow matching: learn v(x, t | c, p) for linear conditional paths.

Noise-to-data (default): x_0 ~ N(0, I), x_1 ~ data, v* = x_1 - x_0.
Data-to-data: x_0 ~ control (p=0) cells, x_1 ~ target perturbation cells (same context),
v* = x_1 - x_0; integrate from provided x_0 via :meth:`sample_from_x0`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torchdiffeq import odeint

from trajectory_replay.model.backbones import (
    DEFAULT_BACKBONE_TYPE,
    build_velocity_backbone,
    normalize_image_shape,
    register_backbone,
    vector_field_module,
    velocity_backbone_in_dim,
)
from trajectory_replay.model.lora import (
    apply_lora_to_module,
    lora_parameters,
    merge_lora_module,
    mlp_has_lora,
)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """t: (batch,) in [0, 1]; returns (batch, dim)."""
    if dim % 2 != 0:
        raise ValueError("time embedding dim must be even")
    half = dim // 2
    device, dtype = t.device, t.dtype
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=device, dtype=dtype) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class DonorFiLMMLP(nn.Module):
    """Shared MLP trunk modulated by a donor embedding at every hidden layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        context_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        self.hidden = nn.ModuleList()
        self.film = nn.ModuleList()
        current = in_dim
        for _ in range(depth - 1):
            self.hidden.append(nn.Linear(current, hidden_dim))
            modulation = nn.Linear(context_dim, 2 * hidden_dim)
            nn.init.zeros_(modulation.weight)
            nn.init.zeros_(modulation.bias)
            self.film.append(modulation)
            current = hidden_dim
        self.output = nn.Linear(current, out_dim)
        self.context_offset = nn.Linear(context_dim, out_dim)
        nn.init.zeros_(self.context_offset.weight)
        nn.init.zeros_(self.context_offset.bias)

    def forward(
        self, features: torch.Tensor, context_embedding: torch.Tensor
    ) -> torch.Tensor:
        h = features
        for layer, modulation in zip(self.hidden, self.film, strict=True):
            scale, shift = modulation(context_embedding).chunk(2, dim=-1)
            h = torch.nn.functional.silu((1.0 + scale) * layer(h) + shift)
        return self.output(h) + self.context_offset(context_embedding)


class ConditionalFlowMatching(nn.Module):
    """Vector field v_theta(x_t, t, c, p) conditioned on context and perturbation embeddings.

    Training target (linear conditional flow matching): x_t = (1-t) x_0 + t x_1 with
    v* = x_1 - x_0. For noise-to-data, x_0 ~ N(0,I); for data-to-data, x_0 is drawn
    from control (p=0) cells and x_1 from the target perturbation (see training utils).

    The vector field backbone (default: flat MLP registered as ``mlp``) is swappable via
    ``backbone_type`` or by passing a pre-built ``backbone`` module. MLP checkpoints
    keep legacy ``mlp.*`` state dict keys.
    """

    def __init__(
        self,
        data_dim: int,
        n_contexts: int,
        n_perturbations: int,
        *,
        cond_emb_dim: int = 32,
        time_emb_dim: int = 64,
        hidden_dim: int = 256,
        depth: int = 4,
        backbone_type: str = DEFAULT_BACKBONE_TYPE,
        backbone: nn.Module | None = None,
        image_shape: tuple[int, ...] | list[int] | None = None,
        unet_num_channels: int = 64,
        unet_num_res_blocks: int = 2,
        unet_attention_resolutions: str = "14",
        unet_dropout: float = 0.0,
        use_conditioning: bool = True,
        covariate_sizes: dict[str, int] | None = None,
        conditioning_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.n_contexts = n_contexts
        self.n_perturbations = n_perturbations
        self.cond_emb_dim = cond_emb_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.backbone_type = backbone_type
        self.use_conditioning = bool(use_conditioning)
        self.conditioning_mode = str(conditioning_mode)
        if self.conditioning_mode not in ("concat", "donor_film"):
            raise ValueError(
                f"conditioning_mode must be 'concat' or 'donor_film', "
                f"got {self.conditioning_mode!r}"
            )
        if self.conditioning_mode == "donor_film" and (
            not self.use_conditioning or backbone_type not in (DEFAULT_BACKBONE_TYPE, "mlp")
        ):
            raise ValueError("donor_film requires a conditional MLP backbone")
        self.covariate_col_order: tuple[str, ...] = tuple(covariate_sizes.keys()) if covariate_sizes else ()
        self.n_covariates = len(self.covariate_col_order)
        self.image_shape: tuple[int, int, int] | None = (
            normalize_image_shape(image_shape) if image_shape is not None else None
        )
        self.unet_num_channels = int(unet_num_channels)
        self.unet_num_res_blocks = int(unet_num_res_blocks)
        self.unet_attention_resolutions = str(unet_attention_resolutions)
        self.unet_dropout = float(unet_dropout)

        if self.use_conditioning:
            self.context_emb = nn.Embedding(n_contexts, cond_emb_dim)
            self.pert_emb = nn.Embedding(n_perturbations, cond_emb_dim)
            if covariate_sizes:
                self.covariate_embs = nn.ModuleDict(
                    {
                        col: nn.Embedding(n_cat, cond_emb_dim)
                        for col, n_cat in covariate_sizes.items()
                    }
                )
            else:
                self.covariate_embs = nn.ModuleDict()
        else:
            self.context_emb = None
            self.pert_emb = None
            self.covariate_embs = nn.ModuleDict()

        if self.conditioning_mode == "donor_film":
            in_dim = data_dim + time_emb_dim + (1 + self.n_covariates) * cond_emb_dim
        else:
            in_dim = velocity_backbone_in_dim(
                data_dim=data_dim,
                time_emb_dim=time_emb_dim,
                cond_emb_dim=cond_emb_dim,
                use_conditioning=self.use_conditioning,
                n_covariates=self.n_covariates,
            )
        if backbone is not None:
            register_backbone(self, backbone, backbone_type)
        elif backbone_type == "unet":
            if self.image_shape is None:
                raise ValueError("image_shape is required when backbone_type='unet'")
            register_backbone(
                self,
                build_velocity_backbone(
                    backbone_type,
                    image_shape=self.image_shape,
                    cond_dim=(2 + self.n_covariates) * cond_emb_dim if self.use_conditioning else 0,
                    unet_num_channels=unet_num_channels,
                    unet_num_res_blocks=unet_num_res_blocks,
                    unet_attention_resolutions=unet_attention_resolutions,
                    unet_dropout=unet_dropout,
                    use_conditioning=self.use_conditioning,
                ),
                backbone_type,
            )
        elif self.conditioning_mode == "donor_film":
            self.mlp = DonorFiLMMLP(
                in_dim,
                data_dim,
                context_dim=cond_emb_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            )
        else:
            register_backbone(
                self,
                build_velocity_backbone(
                    backbone_type,
                    in_dim=in_dim,
                    out_dim=data_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                ),
                backbone_type,
            )

        self.holdout_context: int | None = None
        self.holdout_pert: int | None = None

    def _prepare_z(
        self,
        z: dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        if not self.covariate_col_order:
            return None
        if z is None:
            raise ValueError("covariate tensors required for models with covariates")
        out: dict[str, torch.Tensor] = {}
        for col in self.covariate_col_order:
            t = z[col].to(device=device, dtype=torch.long)
            if t.shape != (batch_size,):
                raise ValueError(f"z[{col!r}] must be shape ({batch_size},), got {tuple(t.shape)}")
            out[col] = t
        return out

    def _covariate_vector(self, z: dict[str, torch.Tensor] | None) -> torch.Tensor:
        if not self.covariate_col_order:
            raise RuntimeError("_covariate_vector called without covariates")
        parts = [self.covariate_embs[col](z[col]) for col in self.covariate_col_order]  # type: ignore[index]
        return torch.cat(parts, dim=-1)

    def _cond_vector(
        self,
        c: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self.use_conditioning:
            raise RuntimeError("_cond_vector called on an unconditional model")
        parts = [self.context_emb(c), self.pert_emb(p)]  # type: ignore[operator]
        if self.covariate_col_order:
            parts.append(self._covariate_vector(z))
        return torch.cat(parts, dim=-1)

    def _flat_to_image(self, x: torch.Tensor) -> torch.Tensor:
        if self.image_shape is None:
            raise ValueError("image_shape is not set on this model")
        c, h, w = self.image_shape
        return x.view(x.shape[0], c, h, w)

    def _image_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    @property
    def uses_spatial_state(self) -> bool:
        """True when the backbone integrates over image-shaped states (UNet)."""
        return self.backbone_type == "unet" and self.image_shape is not None

    def sample_noise(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Draw x_0 ~ N(0, noise_scale^2 I) with the model's native state layout."""
        if self.uses_spatial_state:
            c, h, w = self.image_shape  # type: ignore[misc]
            return noise_scale * torch.randn(
                batch_size, c, h, w, device=device, dtype=dtype, generator=generator
            )
        return noise_scale * torch.randn(
            batch_size, self.data_dim, device=device, dtype=dtype, generator=generator
        )

    def _validate_state(self, x: torch.Tensor, *, name: str = "x") -> None:
        if self.uses_spatial_state:
            c, h, w = self.image_shape  # type: ignore[misc]
            if x.dim() == 4:
                if x.shape[1:] != (c, h, w):
                    raise ValueError(
                        f"{name} must be (B, {c}, {h}, {w}), got {tuple(x.shape)}"
                    )
                return
            if x.dim() == 2 and x.shape[1] == self.data_dim:
                return
            raise ValueError(
                f"{name} must be (B, {self.data_dim}) or (B, {c}, {h}, {w}), got {tuple(x.shape)}"
            )
        if x.dim() != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"{name} must be (B, {self.data_dim}), got {tuple(x.shape)}")

    def _state_to_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_spatial_state and x.dim() == 2:
            return self._flat_to_image(x)
        return x

    def _velocity_output(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return velocity in the same layout as the state ``x`` (flat or spatial)."""
        if self.uses_spatial_state and x.dim() == 4:
            return v
        if self.uses_spatial_state:
            return self._image_to_flat(v)
        return v

    def _velocity_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        te = sinusoidal_time_embedding(t, self.time_emb_dim)
        if not self.use_conditioning:
            return torch.cat([x, te], dim=-1)
        if c is None or p is None:
            raise ValueError("c and p are required for conditional models")
        parts = [x, te, self.context_emb(c), self.pert_emb(p)]  # type: ignore[operator]
        if self.covariate_col_order:
            parts.append(self._covariate_vector(z))
        return torch.cat(parts, dim=-1)

    def _shared_velocity_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        parts = [x, sinusoidal_time_embedding(t, self.time_emb_dim), self.pert_emb(p)]
        if self.covariate_col_order:
            parts.append(self._covariate_vector(z))
        return torch.cat(parts, dim=-1)

    def set_combo_holdout(self, held_context: int, held_pert: int) -> None:
        """Record held-out (context, perturbation) for combo-mode finetuning."""
        self.holdout_context = held_context
        self.holdout_pert = held_pert

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """x: (B, d) or (B, C, H, W); t: (B,). c/p optional and ignored if unconditional."""
        if t.shape != (x.shape[0],):
            raise ValueError("t must be shape (B,)")
        device = x.device
        if self.use_conditioning:
            if c is None or p is None:
                raise ValueError("c and p are required for conditional models")
            if c.shape != (x.shape[0],) or p.shape != (x.shape[0],):
                raise ValueError("c and p must be shape (B,)")
            z = self._prepare_z(z, x.shape[0], device)

        if self.backbone_type == "unet":
            if x.dim() == 2:
                if x.shape[1] != self.data_dim:
                    raise ValueError(f"x must be (B, {self.data_dim}), got {tuple(x.shape)}")
                x_img = self._flat_to_image(x)
            elif x.dim() == 4:
                x_img = x
            else:
                raise ValueError(f"unet expects x as (B, {self.data_dim}) or (B, C, H, W)")
            cond = self._cond_vector(c, p, z) if self.use_conditioning else None
            v_img = vector_field_module(self)(x_img, t, cond)
            return self._velocity_output(v_img, x)

        if x.dim() != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must be (B, {self.data_dim}), got {tuple(x.shape)}")
        if self.conditioning_mode == "donor_film":
            shared = self._shared_velocity_features(x, t, p, z)  # type: ignore[arg-type]
            context = self.context_emb(c)  # type: ignore[operator]
            return self.mlp(shared, context)
        features = self._velocity_features(x, t, c, p, z)
        return vector_field_module(self)(features)

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
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        options: dict[str, Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate dx/dt = v_theta(x, t) from t=0 to ``t_end``; x_0 ~ N(0, noise_scale^2 I).

        For conditional models, pass ``c`` and ``p`` (batch size inferred from them).
        For unconditional models, pass ``batch_size`` (or ``c`` only to infer batch size).
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if self.use_conditioning:
            if c is None or p is None:
                raise ValueError("c and p are required for conditional sample()")
            c = c.to(device=device, dtype=torch.long)
            p = p.to(device=device, dtype=torch.long)
            b = c.shape[0]
        else:
            if batch_size is not None:
                b = int(batch_size)
            elif c is not None:
                b = c.shape[0]
            else:
                raise ValueError("batch_size or c must be provided for unconditional sample()")
            c = None
            p = None
        x0 = self.sample_noise(
            b,
            device=device,
            dtype=dtype,
            generator=generator,
            noise_scale=noise_scale,
        )
        return self.sample_from_x0(
            x0,
            c,
            p,
            z=z,
            t_end=t_end,
            method=method,
            rtol=rtol,
            atol=atol,
            options=options,
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
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Integrate from given x_0 to time ``t_end`` in [0, 1]."""
        if not (0.0 < t_end <= 1.0):
            raise ValueError(f"t_end must be in (0, 1], got {t_end}")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self._validate_state(x0, name="x0")
        b = x0.shape[0]
        if self.use_conditioning:
            if c is None or p is None:
                raise ValueError("c and p are required for conditional sample_from_x0()")
            c = c.to(device=device, dtype=torch.long)
            p = p.to(device=device, dtype=torch.long)
            if c.shape != (b,) or p.shape != (b,):
                raise ValueError("c and p must be shape (B,) matching x0 batch size")
            z = self._prepare_z(z, b, device)
        else:
            c = None
            p = None
            z = None
        x_init = self._state_to_backbone_input(x0.to(device=device, dtype=dtype))
        t_span = torch.tensor([0.0, float(t_end)], device=device, dtype=dtype)

        def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            tb = torch.full((x.shape[0],), t.item(), device=x.device, dtype=dtype)
            return self.forward(x, tb, c, p, z)

        sol = odeint(
            ode_func,
            x_init,
            t_span,
            rtol=rtol,
            atol=atol,
            method=method,
            options=options,
        )
        out = sol[-1]
        if self.uses_spatial_state and x0.dim() == 2:
            return self._image_to_flat(out)
        return out

    def expand_perturbation_embedding(
        self,
        insert_at: int,
        *,
        init_std: float = 0.02,
    ) -> None:
        """Grow ``pert_emb`` by one row at ``insert_at`` (held-out index); copy existing rows."""
        if not self.use_conditioning or self.pert_emb is None:
            raise RuntimeError("expand_perturbation_embedding requires use_conditioning=True")
        if insert_at < 0 or insert_at > self.n_perturbations:
            raise ValueError(f"insert_at must be in [0, {self.n_perturbations}], got {insert_at}")
        old = self.pert_emb
        n_old, dim = old.num_embeddings, old.embedding_dim
        device, dtype = old.weight.device, old.weight.dtype
        new_emb = nn.Embedding(n_old + 1, dim, device=device, dtype=dtype)
        with torch.no_grad():
            if insert_at > 0:
                new_emb.weight[:insert_at].copy_(old.weight[:insert_at])
            nn.init.normal_(new_emb.weight[insert_at : insert_at + 1], std=init_std)
            if insert_at < n_old:
                new_emb.weight[insert_at + 1 :].copy_(old.weight[insert_at:])
        self.pert_emb = new_emb
        self.n_perturbations = n_old + 1

    def expand_context_embedding(
        self,
        insert_at: int,
        *,
        init_std: float = 0.02,
    ) -> None:
        """Grow ``context_emb`` by one row at ``insert_at`` (held-out index); copy existing rows."""
        if not self.use_conditioning or self.context_emb is None:
            raise RuntimeError("expand_context_embedding requires use_conditioning=True")
        if insert_at < 0 or insert_at > self.n_contexts:
            raise ValueError(f"insert_at must be in [0, {self.n_contexts}], got {insert_at}")
        old = self.context_emb
        n_old, dim = old.num_embeddings, old.embedding_dim
        device, dtype = old.weight.device, old.weight.dtype
        new_emb = nn.Embedding(n_old + 1, dim, device=device, dtype=dtype)
        with torch.no_grad():
            if insert_at > 0:
                new_emb.weight[:insert_at].copy_(old.weight[:insert_at])
            nn.init.normal_(new_emb.weight[insert_at : insert_at + 1], std=init_std)
            if insert_at < n_old:
                new_emb.weight[insert_at + 1 :].copy_(old.weight[insert_at:])
        self.context_emb = new_emb
        self.n_contexts = n_old + 1

    def apply_mlp_lora(
        self,
        *,
        rank: int,
        alpha: float,
        train_output_layer: bool = False,
    ) -> None:
        """Wrap MLP ``Linear`` layers with LoRA adapters (base weights stay frozen)."""
        if self.backbone_type not in (DEFAULT_BACKBONE_TYPE, "mlp", "residual_mlp"):
            raise RuntimeError(f"LoRA is only supported for mlp backbones, got {self.backbone_type!r}")
        vf = vector_field_module(self)
        if mlp_has_lora(vf):
            raise RuntimeError("MLP already has LoRA adapters")
        apply_lora_to_module(
            vf,
            rank=rank,
            alpha=alpha,
            train_output_layer=train_output_layer,
        )

    def merge_mlp_lora(self) -> None:
        """Fold LoRA deltas into base ``Linear`` weights and drop adapter modules."""
        if self.backbone_type not in (DEFAULT_BACKBONE_TYPE, "mlp", "residual_mlp"):
            raise RuntimeError(f"LoRA merge is only supported for mlp backbones, got {self.backbone_type!r}")
        merge_lora_module(vector_field_module(self))

    def configure_finetune(
        self,
        *,
        holdout_mode: str,
        train_vector_field: bool,
        train_lora: bool = False,
        train_old_pert_embeddings: bool = False,
        train_old_context_embeddings: bool = False,
        train_covariate_embeddings: bool = False,
        held_out_pert: int,
        held_out_context: int | None = None,
        train_old_embeddings: bool | None = None,
    ) -> tuple[list[nn.Parameter], int | None, int | None]:
        """Return (optimizer params, optional pert/context row indices for grad masking).

        ``perturbation`` mode: held-out perturbation row is always trainable (others optional).
        ``context`` mode: held-out context row is always trainable (others optional).
        ``combo`` mode: held-out context and perturbation rows are trainable unless the
        corresponding table is fully unfrozen via ``train_old_*_embeddings``.

        ``train_old_embeddings`` (deprecated) sets both ``train_old_pert_embeddings`` and
        ``train_old_context_embeddings`` when provided.
        """
        if train_vector_field and train_lora:
            raise ValueError("train_vector_field and train_lora are mutually exclusive")
        vf = vector_field_module(self)
        lora_backbone = self.backbone_type in (
            DEFAULT_BACKBONE_TYPE,
            "mlp",
            "residual_mlp",
        )
        if train_lora and lora_backbone and not mlp_has_lora(vf):
            raise ValueError("train_lora=True requires apply_mlp_lora() on the model first")

        if train_old_embeddings is not None:
            train_old_pert_embeddings = train_old_embeddings
            train_old_context_embeddings = train_old_embeddings

        if not self.use_conditioning:
            raise RuntimeError("configure_finetune requires use_conditioning=True")

        for param in self.parameters():
            param.requires_grad = False

        trainable: list[nn.Parameter] = []
        if train_vector_field:
            for param in vf.parameters():
                param.requires_grad = True
                trainable.append(param)
        elif train_lora and lora_backbone:
            for param in lora_parameters(vf):
                param.requires_grad = True
                trainable.append(param)

        if train_covariate_embeddings:
            for param in self.covariate_embs.parameters():
                param.requires_grad = True
                trainable.append(param)

        pert_mask = None if train_old_pert_embeddings else held_out_pert
        context_mask = (
            None if train_old_context_embeddings else held_out_context
        )

        if holdout_mode == "combo":
            if held_out_context is None:
                raise ValueError("held_out_context is required for combo holdout_mode")
            self.set_combo_holdout(held_out_context, held_out_pert)
            self.context_emb.weight.requires_grad = True
            self.pert_emb.weight.requires_grad = True
            trainable.append(self.context_emb.weight)
            trainable.append(self.pert_emb.weight)
            return trainable, pert_mask, context_mask

        if holdout_mode == "context":
            if held_out_context is None:
                raise ValueError("held_out_context is required for context holdout_mode")
            self.holdout_context = held_out_context
            self.context_emb.weight.requires_grad = True
            trainable.append(self.context_emb.weight)
            if train_old_pert_embeddings:
                self.pert_emb.weight.requires_grad = True
                trainable.append(self.pert_emb.weight)
            return trainable, pert_mask, context_mask

        if holdout_mode != "perturbation":
            raise ValueError(f"Unknown holdout_mode {holdout_mode!r}")

        self.pert_emb.weight.requires_grad = True
        trainable.append(self.pert_emb.weight)
        if train_old_context_embeddings:
            self.context_emb.weight.requires_grad = True
            trainable.append(self.context_emb.weight)
        return trainable, pert_mask, context_mask

    @staticmethod
    def mask_embedding_rows_grad(
        weight: torch.Tensor, train_rows: frozenset[int]
    ) -> None:
        """Zero gradients on embedding rows not listed in ``train_rows``."""
        grad = weight.grad
        if grad is None:
            return
        with torch.no_grad():
            for i in range(grad.shape[0]):
                if i not in train_rows:
                    grad[i].zero_()

    @staticmethod
    def mask_embedding_row_grad(weight: torch.Tensor, train_index: int) -> None:
        """Zero gradients on embedding rows other than ``train_index``."""
        ConditionalFlowMatching.mask_embedding_rows_grad(
            weight, frozenset({train_index})
        )


def resolve_use_conditioning_from_config(cfg: dict[str, Any]) -> bool:
    """Read ``use_conditioning`` from a checkpoint config (legacy keys supported)."""
    if "use_conditioning" in cfg:
        return bool(cfg["use_conditioning"])
    conditioning = cfg.get("conditioning", "conditional")
    return str(conditioning) != "unconditional"


def build_conditional_flow_matching_from_config(cfg: dict[str, Any]) -> ConditionalFlowMatching:
    """Reconstruct a model from a saved checkpoint config dict."""
    backbone_type = str(cfg.get("backbone_type", DEFAULT_BACKBONE_TYPE))
    image_shape = cfg.get("image_shape")
    covariate_sizes = cfg.get("covariate_sizes")
    if covariate_sizes is None and "encoding" in cfg:
        enc = cfg["encoding"]
        order = tuple(enc.get("covariate_col_order") or ())
        covariate_sizes = {
            col: len(enc["covariate_categories"][col]) for col in order
        }
    return ConditionalFlowMatching(
        data_dim=int(cfg["data_dim"]),
        n_contexts=int(cfg["n_contexts"]),
        n_perturbations=int(cfg["n_perturbations"]),
        cond_emb_dim=int(cfg["cond_emb_dim"]),
        time_emb_dim=int(cfg["time_emb_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        backbone_type=backbone_type,
        image_shape=image_shape,
        unet_num_channels=int(cfg.get("unet_num_channels", 64)),
        unet_num_res_blocks=int(cfg.get("unet_num_res_blocks", 2)),
        unet_attention_resolutions=str(cfg.get("unet_attention_resolutions", "14")),
        unet_dropout=float(cfg.get("unet_dropout", 0.0)),
        use_conditioning=resolve_use_conditioning_from_config(cfg),
        covariate_sizes=dict(covariate_sizes) if covariate_sizes else None,
        conditioning_mode=str(cfg.get("conditioning_mode", "concat")),
    )
