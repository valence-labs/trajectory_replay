#!/usr/bin/env python3
"""Render ``*_gen.pt`` sample tensors to PNG grids (sibling ``.png`` files).

Expects tensors in ``[-1, 1]`` with shape ``(N, C, H, W)`` — the format written
by ``train_cl.py``.

Example:
  uv run python experiments/image_cl/render_sample_grids.py \\
    --root results_dir/image_cl/cifar10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def render_one(pt_path: Path, *, nrow: int, overwrite: bool) -> Path | None:
    out = pt_path.with_suffix(".png")
    if out.is_file() and not overwrite:
        return None
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("x", "samples", "gens", "images"):
            if key in raw and torch.is_tensor(raw[key]):
                x = raw[key]
                break
        else:
            raise ValueError(f"{pt_path}: dict with no sample tensor key")
    elif torch.is_tensor(raw):
        x = raw
    else:
        raise TypeError(f"{pt_path}: unsupported payload {type(raw)}")
    if x.ndim != 4:
        raise ValueError(f"{pt_path}: expected NCHW, got {tuple(x.shape)}")
    x = x.detach().float().cpu()
    # Stored in [-1, 1]; clamp for display.
    x = x.clamp(-1.0, 1.0)
    grid = make_grid(x, nrow=nrow, normalize=True, value_range=(-1.0, 1.0), padding=2)
    save_image(grid, out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results_dir/image_cl/cifar10"),
        help="Directory to walk for *_gen.pt files.",
    )
    p.add_argument("--nrow", type=int, default=8, help="Images per grid row.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--glob",
        default="**/*_gen.pt",
        help="Glob under --root (default: **/ *_gen.pt).",
    )
    args = p.parse_args()
    root = args.root.resolve()
    paths = sorted(root.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matching {args.glob!r} under {root}")
    written = skipped = 0
    for pt in paths:
        out = render_one(pt, nrow=args.nrow, overwrite=args.overwrite)
        if out is None:
            skipped += 1
        else:
            written += 1
            print(f"  wrote {out.relative_to(root)}")
    print(f"Done: wrote {written}, skipped {skipped} (already present), scanned {len(paths)}")


if __name__ == "__main__":
    main()
