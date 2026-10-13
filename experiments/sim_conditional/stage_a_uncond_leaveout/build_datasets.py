#!/usr/bin/env python3
"""Build Stage A leave-out MoG data under the unconditional sibling tree.

Geometry matches ``stage_a_leaveout`` (same defaults). Writes to
``results_dir/sim_conditional/stage_a_uncond_leaveout/data/dim<d>/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.sim_conditional.common import stage_a_data_root
from experiments.sim_conditional.stage_a_leaveout import build_datasets as cond_build
from experiments.sim_conditional.stage_a_uncond_leaveout.common import (
    REPO_ROOT,
    stage_a_uncond_data_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument(
        "--reuse-conditional",
        action="store_true",
        help="Copy classes.npz/meta from stage_a_leaveout if present (same geometry).",
    )
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--heldout-class", type=int, default=None)
    p.add_argument("--separation", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--n-train", type=int, default=8_000)
    p.add_argument("--n-val", type=int, default=1_000)
    p.add_argument("--n-test", type=int, default=1_000)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or stage_a_uncond_data_root(REPO_ROOT, dim=args.dim)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = stage_a_data_root(REPO_ROOT, dim=args.dim)
    if args.reuse_conditional and (src / "classes.npz").is_file():
        shutil.copy2(src / "classes.npz", out_dir / "classes.npz")
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        meta["stage"] = "a_uncond_leaveout"
        meta["source"] = str(src)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Copied Stage A data from {src} → {out_dir}")
        return

    # Delegate to conditional builder with our out-dir, then retag meta.
    cond_args = argparse.Namespace(
        dim=args.dim,
        n_classes=args.n_classes,
        heldout_class=args.heldout_class,
        separation=args.separation,
        sigma=args.sigma,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        data_seed=args.data_seed,
        out_dir=out_dir,
    )
    # Call builder main body by temporarily patching argv is messy; invoke internals.
    # Re-exec style: set sys.argv and call cond_build.main
    import sys

    argv = [
        "build_datasets.py",
        "--dim",
        str(args.dim),
        "--n-classes",
        str(args.n_classes),
        "--separation",
        str(args.separation),
        "--sigma",
        str(args.sigma),
        "--n-train",
        str(args.n_train),
        "--n-val",
        str(args.n_val),
        "--n-test",
        str(args.n_test),
        "--data-seed",
        str(args.data_seed),
        "--out-dir",
        str(out_dir),
    ]
    if args.heldout_class is not None:
        argv.extend(["--heldout-class", str(args.heldout_class)])
    old = sys.argv
    try:
        sys.argv = argv
        cond_build.main()
    finally:
        sys.argv = old

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["stage"] = "a_uncond_leaveout"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Tagged meta stage=a_uncond_leaveout under {out_dir}")


if __name__ == "__main__":
    main()
