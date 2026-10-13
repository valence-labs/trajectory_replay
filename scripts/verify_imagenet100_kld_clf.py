#!/usr/bin/env python3
"""GPU smoke check for the ImageNet-100 ResNet50-V2 stream-KLD probe.

Decodes a slice of cached val latents with the frozen SD-VAE, classifies with
CMC-100 masked argmax, and prints top-1 accuracy. Run on a compute GPU node::

    sbatch sbatch/image_cl/verify_imagenet100_kld.sbatch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--latent-dir",
        type=Path,
        default=Path.home()
        / "sandbox/tx_analysis_data/imagenet100/latents_sdvae_ft_mse_256",
    )
    p.add_argument("--vae-id", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    from trajectory_replay.eval.stream_kld import (
        load_imagenet100_kld_classifier,
        load_imagenet100_synsets_from_latent_dir,
        load_imagenet2012_wnids,
    )
    from trajectory_replay.model.sd_vae import FrozenSDVAE

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable — refuse to run on CPU login nodes")
    device = torch.device(args.device)
    print(f"device={device}  latent_dir={args.latent_dir}")

    wnids = load_imagenet2012_wnids()
    assert len(wnids) == 1000 and wnids[0] == "n01440764"
    synsets = load_imagenet100_synsets_from_latent_dir(args.latent_dir)
    clf = load_imagenet100_kld_classifier(synsets, device=device)
    print(
        f"synsets={len(synsets)}  global_idx[:5]={clf.global_indices[:5].tolist()}  "
        f"synsets[:5]={synsets[:5]}"
    )

    blob = torch.load(args.latent_dir / "val.pt", map_location="cpu", weights_only=False)
    n = min(int(args.n), int(blob["z"].shape[0]))
    z = blob["z"][:n].to(dtype=torch.float32)
    y = blob["y"][:n].long()
    print(f"eval n={n}  z={tuple(z.shape)}")

    vae = FrozenSDVAE.from_pretrained(args.vae_id, device=device)
    pix = vae.decode(z.to(device), batch_size=int(args.batch_size)).cpu()
    print(f"decoded pix={tuple(pix.shape)} range=[{float(pix.min()):.3f}, {float(pix.max()):.3f}]")

    pred = clf.predict_local(pix, device=device, batch_size=int(args.batch_size))
    acc = float((pred == y).float().mean())
    print(f"top1 on decoded val: {acc:.4f}  ({int((pred == y).sum())}/{n})")
    print(f"pred[:16]={pred[:16].tolist()}")
    print(f"true[:16]={y[:16].tolist()}")
    if acc < 0.5:
        raise SystemExit(
            f"KLD probe accuracy {acc:.3f} looks broken "
            "(expected >>0.5 on real SD-VAE reconstructions of CMC-100)."
        )
    print("OK")


if __name__ == "__main__":
    main()
