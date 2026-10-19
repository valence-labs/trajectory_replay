#!/usr/bin/env python3
"""Create ImageNet-100 train/val trees by symlinking selected synsets from a full ImageNet root.

Source layout (torchvision)::

    <imagenet-root>/train/<synset>/*.JPEG
    <imagenet-root>/val/<synset>/*.JPEG

Destination::

    <out-root>/train/<synset>  -> symlink to source class dir
    <out-root>/val/<synset>    -> symlink to source class dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_dir" / "imagenet",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_dir" / "imagenet100",
    )
    p.add_argument(
        "--synsets-file",
        type=Path,
        default=Path("src/trajectory_replay/data/imagenet100_cmc.txt"),
    )
    p.add_argument("--force", action="store_true", help="Replace existing symlinks")
    args = p.parse_args()

    synsets = [ln.strip() for ln in args.synsets_file.read_text().splitlines() if ln.strip()]
    if len(synsets) != 100:
        print(f"expected 100 synsets, got {len(synsets)}", file=sys.stderr)
        return 1

    missing = []
    linked = 0
    for split in ("train", "val"):
        src_split = args.imagenet_root / split
        dst_split = args.out_root / split
        dst_split.mkdir(parents=True, exist_ok=True)
        for syn in synsets:
            src = src_split / syn
            dst = dst_split / syn
            if not src.is_dir():
                missing.append(f"{split}/{syn}")
                continue
            if dst.is_symlink() or dst.exists():
                if args.force:
                    if dst.is_symlink() or dst.is_file():
                        dst.unlink()
                    else:
                        print(f"refusing to replace non-symlink {dst}", file=sys.stderr)
                        return 1
                else:
                    linked += 1
                    continue
            dst.symlink_to(src.resolve())
            linked += 1

    print(f"out_root={args.out_root}")
    print(f"linked_or_existing={linked} missing={len(missing)}")
    if missing:
        print("missing examples:", missing[:10])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
