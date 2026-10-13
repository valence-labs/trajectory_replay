#!/usr/bin/env python3
"""Encode ImageNet (torchvision layout) to frozen SD-VAE latents.

Reads::

    <image-root>/train/<synset>/*.JPEG
    <image-root>/val/<synset>/*.JPEG

Writes::

    <out-dir>/meta.json
    <out-dir>/train.pt
    <out-dir>/val.pt

Each ``*.pt`` is a dict::

    {
      "z": FloatTensor[N, 4, 32, 32],   # posterior mean * scaling_factor
      "y": LongTensor[N],              # class index into ``synsets``
      "synsets": list[str],            # ImageFolder class-folder order
      "image_size": 256,
      "vae_id": "stabilityai/sd-vae-ft-mse",
      "scaling_factor": float,
      "latent_stats": "mean",          # not sampled
    }

Pixels are resized/center-cropped to ``--image-size`` and normalized to ``[-1, 1]``
(SD / DiT convention) before encoding. The VAE stays frozen; only the encoder runs.

Optional ``--save-decode-samples N`` writes a small orig|recon PNG grid per split
for visual QA (top row = input crop, bottom row = VAE decode of cached latents).

Default target is CMC ImageNet-100 under ``~/sandbox/tx_analysis_data/imagenet100``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _load_first_n_images(split_dir: Path, image_size: int, n: int) -> torch.Tensor:
    """Return ``(n, 3, H, W)`` in ``[-1, 1]`` (ImageFolder order)."""
    ds = ImageFolder(str(split_dir), transform=_build_transform(image_size))
    n = min(n, len(ds))
    xs = [ds[i][0] for i in range(n)]
    return torch.stack(xs, dim=0)


@torch.no_grad()
def _save_decode_samples(
    *,
    split: str,
    split_dir: Path,
    blob: dict,
    vae: torch.nn.Module,
    device: torch.device,
    image_size: int,
    n_samples: int,
    out_path: Path,
) -> Path:
    """Write a 2-row grid: originals (top) vs reconstructions (bottom)."""
    n = min(int(n_samples), int(blob["n"]), int(blob["z"].shape[0]))
    if n <= 0:
        raise ValueError("n_samples must be > 0")

    x = _load_first_n_images(split_dir, image_size, n)
    scaling = float(blob["scaling_factor"])
    z = blob["z"][:n].to(device=device, dtype=torch.float32) / scaling

    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        recon = vae.decode(z).sample
    recon = recon.detach().float().cpu().clamp(-1.0, 1.0)

    orig_01 = (x + 1.0) * 0.5
    recon_01 = (recon + 1.0) * 0.5
    grid = make_grid(torch.cat([orig_01, recon_01], dim=0), nrow=n, padding=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))
    print(f"wrote decode samples ({split}, n={n}): {out_path}", flush=True)
    return out_path


@torch.no_grad()
def _encode_split(
    *,
    split_dir: Path,
    vae: torch.nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    out_dtype: torch.dtype,
    scaling_factor: float,
) -> dict:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split dir: {split_dir}")

    ds = ImageFolder(str(split_dir), transform=_build_transform(image_size))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    zs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in tqdm(loader, desc=split_dir.name, leave=True):
        x = x.to(device, non_blocking=True)
        # SD-VAE expects float32 activations; autocast speeds the encoder on GPU.
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            posterior = vae.encode(x).latent_dist
            z = posterior.mean * scaling_factor
        zs.append(z.detach().to(dtype=out_dtype, device="cpu"))
        ys.append(y.cpu().long())

    z_all = torch.cat(zs, dim=0)
    y_all = torch.cat(ys, dim=0)
    synsets = list(ds.classes)
    if z_all.shape[0] != len(ds):
        raise RuntimeError(f"row mismatch: z={z_all.shape[0]} ds={len(ds)}")
    return {
        "z": z_all,
        "y": y_all,
        "synsets": synsets,
        "image_size": int(image_size),
        "scaling_factor": float(scaling_factor),
        "latent_stats": "mean",
        "n": int(z_all.shape[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image-root",
        type=Path,
        default=Path.home() / "sandbox/tx_analysis_data/imagenet100",
        help="Torchvision ImageNet root with train/ and val/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for meta.json + {train,val}.pt "
        "(default: <image-root>/latents_sdvae_ft_mse_<image-size>)",
    )
    p.add_argument("--vae-id", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=("train", "val"),
    )
    p.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="On-disk latent dtype (cast to float32 at train time).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode even if split .pt already exists.",
    )
    p.add_argument(
        "--save-decode-samples",
        type=int,
        default=0,
        metavar="N",
        help="If N>0, save an orig|recon PNG grid with N images per split "
        "(uses existing .pt when present; does not require --overwrite).",
    )
    args = p.parse_args()

    image_root = args.image_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (image_root / f"latents_sdvae_ft_mse_{args.image_size}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = torch.device(args.device)

    print(f"image_root={image_root}")
    print(f"out_dir={out_dir}")
    print(f"vae_id={args.vae_id} device={device} dtype={args.dtype}")
    print(f"image_size={args.image_size} batch_size={args.batch_size}")
    print(f"save_decode_samples={args.save_decode_samples}")

    from diffusers import AutoencoderKL

    print(f"loading VAE {args.vae_id} …", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_id)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    scaling = float(vae.config.scaling_factor)
    print(f"scaling_factor={scaling}", flush=True)

    summary: dict = {
        "image_root": str(image_root),
        "out_dir": str(out_dir),
        "vae_id": args.vae_id,
        "image_size": args.image_size,
        "scaling_factor": scaling,
        "latent_stats": "mean",
        "dtype": args.dtype,
        "save_decode_samples": int(args.save_decode_samples),
        "splits": {},
    }

    for split in args.splits:
        out_pt = out_dir / f"{split}.pt"
        skipped = False
        if out_pt.is_file() and not args.overwrite:
            print(f"skip {split}: exists {out_pt}", flush=True)
            blob = torch.load(out_pt, map_location="cpu", weights_only=False)
            skipped = True
        else:
            blob = _encode_split(
                split_dir=image_root / split,
                vae=vae,
                device=device,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dtype=out_dtype,
                scaling_factor=scaling,
            )
            blob["vae_id"] = args.vae_id
            torch.save(blob, out_pt)
            print(
                f"wrote {out_pt} z={tuple(blob['z'].shape)} classes={len(blob['synsets'])}",
                flush=True,
            )

        split_meta = {
            "path": str(out_pt),
            "n": int(blob["n"]),
            "n_classes": len(blob["synsets"]),
            "z_shape": list(blob["z"].shape),
            "skipped": skipped,
        }

        if args.save_decode_samples > 0:
            sample_path = out_dir / f"decode_samples_{split}.png"
            _save_decode_samples(
                split=split,
                split_dir=image_root / split,
                blob=blob,
                vae=vae,
                device=device,
                image_size=int(blob.get("image_size", args.image_size)),
                n_samples=args.save_decode_samples,
                out_path=sample_path,
            )
            split_meta["decode_samples"] = str(sample_path)

        summary["splits"][split] = split_meta

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — surface full error in sbatch logs
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
