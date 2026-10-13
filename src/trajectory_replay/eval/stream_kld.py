"""Masip-style stream class-histogram KLD / ARD for unconditional image generators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from trajectory_replay.eval.image_fid import ToPixelsFn

# Alphabetical WNID order = torchvision / timm ImageNet-1k logit index order.
_IMAGENET2012_WNIDS_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "imagenet2012_wnids.txt"
)


def load_imagenet2012_wnids(path: str | Path | None = None) -> list[str]:
    """Return the 1000 ImageNet-2012 WNIDs in torchvision logit order."""
    wnids_path = Path(path) if path is not None else _IMAGENET2012_WNIDS_FILE
    wnids = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
    if len(wnids) != 1000:
        raise ValueError(f"expected 1000 WNIDs in {wnids_path}, found {len(wnids)}")
    return wnids


def load_fmnist_kld_classifier(
    weights_dir: str | Path,
    *,
    device: torch.device,
    num_classes: int = 10,
) -> nn.Module:
    """Load the ResNet18 classifier used by Masip for FMNIST class-KLD."""
    from torchvision.models import resnet18

    weights_dir = Path(weights_dir)
    ckpt = weights_dir / "resnet.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"KLD classifier weights not found: {ckpt}. "
            "Train with baselines/diffusion_continual_learning/train_kld_classifier.py "
            "or point --kld-clf-path at the Masip weights/cnn_fmnist directory."
        )
    clf = resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, num_classes)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    clf.load_state_dict(state)
    clf.to(device)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


@dataclass
class ImageNet100KLDClassifier:
    """Frozen ResNet50-V2 probe with CMC-100 masked argmax → local ``0..99`` labels.

    Logits outside the cached ``synsets`` are ignored before argmax, then the
    winning CMC index is mapped back to the local ImageFolder / latent label
    space (same order as ``synsets`` in the latent blob).
    """

    model: nn.Module
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    global_indices: torch.Tensor  # (100,) int64 ImageNet-1k logit indices
    synsets: list[str]
    weights_name: str = "IMAGENET1K_V2"
    num_classes: int = 100

    @torch.no_grad()
    def predict_local(
        self,
        images: torch.Tensor,
        *,
        device: torch.device,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Classify RGB images in ``[-1, 1]`` or ``[0, 1]`` → local class ids."""
        preds: list[torch.Tensor] = []
        idx = self.global_indices.to(device=device, dtype=torch.long)
        for start in range(0, images.shape[0], batch_size):
            chunk = _to_01_rgb(images[start : start + batch_size].to(device))
            # V2 transforms expect float RGB in [0, 1] (see convert_image_dtype).
            x = self.preprocess(chunk)
            logits = self.model(x)
            masked = logits.index_select(dim=1, index=idx)
            local = torch.argmax(masked, dim=1)
            preds.append(local.cpu())
        return torch.cat(preds, dim=0)


def load_imagenet100_kld_classifier(
    synsets: list[str],
    *,
    device: torch.device,
    weights_name: str = "IMAGENET1K_V2",
) -> ImageNet100KLDClassifier:
    """Load torchvision ResNet50 and build the CMC-100 logit mask from ``synsets``."""
    from torchvision.models import ResNet50_Weights, resnet50

    if len(synsets) != 100:
        raise ValueError(f"expected 100 ImageNet-100 synsets, got {len(synsets)}")
    if len(set(synsets)) != 100:
        raise ValueError("ImageNet-100 synsets must be unique")

    wnids = load_imagenet2012_wnids()
    wnid_to_global = {wnid: i for i, wnid in enumerate(wnids)}
    missing = [s for s in synsets if s not in wnid_to_global]
    if missing:
        raise ValueError(
            f"{len(missing)} synsets not in ImageNet-2012, e.g. {missing[:5]}"
        )
    global_indices = torch.tensor(
        [wnid_to_global[s] for s in synsets], dtype=torch.long
    )

    try:
        weights = ResNet50_Weights[weights_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ResNet50 weights {weights_name!r}; "
            f"use one of {[w.name for w in ResNet50_Weights]}"
        ) from exc

    print(f"Loading ImageNet-100 KLD classifier: resnet50({weights_name})")
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return ImageNet100KLDClassifier(
        model=model,
        preprocess=weights.transforms(),
        global_indices=global_indices,
        synsets=list(synsets),
        weights_name=weights_name,
        num_classes=len(synsets),
    )


def load_imagenet100_synsets_from_latent_dir(latent_dir: str | Path) -> list[str]:
    """Read ``synsets`` from ``val.pt`` (smaller than ``train.pt``)."""
    path = Path(latent_dir) / "val.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; needed for ImageNet-100 KLD synset→logit mapping."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    synsets = list(blob["synsets"])
    if len(synsets) != 100:
        raise ValueError(f"expected 100 synsets in {path}, found {len(synsets)}")
    return synsets


def _to_01_rgb(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if x.min() < 0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


@torch.no_grad()
def classify_images(
    clf: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Return predicted class ids. ``images`` in [-1,1] or [0,1], C=1 or 3.

    Uses Masip's ``(x - 0.5) * 2`` convention for the custom FMNIST/CIFAR
    ResNet18 checkpoints — **not** for pretrained torchvision ImageNet models.
    """
    preds: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        chunk = _to_01_rgb(images[start : start + batch_size].to(device))
        # Masip: classifier sees (x - 0.5) * 2 with x in [0, 1].
        logits = clf((chunk - 0.5) * 2.0)
        preds.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(preds, dim=0)


def histogram_kld_ard(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Class-histogram KLD / ARD matching Masip ``DistributionMetrics``."""
    y_true = y_true.detach().cpu().long().view(-1)
    y_pred = y_pred.detach().cpu().long().view(-1)
    hist_true = torch.zeros(num_classes, dtype=torch.float64)
    hist_pred = torch.zeros(num_classes, dtype=torch.float64)
    for i in range(num_classes):
        hist_true[i] = torch.sum(y_true == i)
        hist_pred[i] = torch.sum(y_pred == i)
    ratio_true = hist_true / hist_true.sum().clamp_min(1.0)
    ratio_pred = hist_pred / hist_pred.sum().clamp_min(1.0)
    ard = float(torch.sum(torch.abs(ratio_true - ratio_pred)).item() / 2.0)
    kl = nn.KLDivLoss(reduction="batchmean")
    kld = float(
        kl(
            (ratio_pred + 1e-8).log().unsqueeze(0),
            (ratio_true + 1e-8).unsqueeze(0),
        ).item()
    )
    return {
        "stream_kld": kld,
        "stream_ard": ard,
        "hist_true": ratio_true.tolist(),
        "hist_pred": ratio_pred.tolist(),
    }


@torch.no_grad()
def stream_kld_from_model(
    model: nn.Module,
    y_true_stream: torch.Tensor,
    *,
    clf: nn.Module | ImageNet100KLDClassifier,
    device: torch.device,
    n_gen: int,
    batch_size: int = 128,
    ode_method: str = "euler",
    ode_step_size: float = 0.1,
    mean_flow_n_steps: int | None = None,
    seed: int = 0,
    num_classes: int | None = None,
    to_pixels: ToPixelsFn | None = None,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    """Generate samples, classify, compare class histogram to ``y_true_stream``.

    For latent FM, pass ``to_pixels`` so ODE latents are decoded before the
    classifier (same convention as FID).

    Class-conditional models sample labels from the empirical distribution of
    ``y_true_stream`` so KLD measures classifier agreement under matched class
    priors (not label-sampling mismatch).
    """
    from trajectory_replay.eval.image_fid import generate_samples

    use_cond = bool(getattr(model, "use_conditioning", False))
    fake = generate_samples(
        model,
        n_gen,
        device=device,
        batch_size=batch_size,
        ode_method=ode_method,
        ode_step_size=ode_step_size,
        mean_flow_n_steps=mean_flow_n_steps,
        seed=seed,
        to_pixels=to_pixels,
        label_source=y_true_stream if use_cond else None,
        guidance_scale=guidance_scale,
    )
    if isinstance(clf, ImageNet100KLDClassifier):
        y_pred = clf.predict_local(fake, device=device, batch_size=batch_size)
        n_cls = int(clf.num_classes if num_classes is None else num_classes)
    else:
        y_pred = classify_images(clf, fake, device=device, batch_size=batch_size)
        n_cls = 10 if num_classes is None else int(num_classes)
    out = histogram_kld_ard(y_true_stream, y_pred, num_classes=n_cls)
    out["n_gen_kld"] = int(n_gen)
    out["n_true_labels"] = int(y_true_stream.numel())
    return out
