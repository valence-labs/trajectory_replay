#!/usr/bin/env python3
"""Convert HuggingFace ILSVRC/imagenet-1k parquet shards to torchvision ImageNet layout.

Writes::

    <out-root>/
      train/<synset>/*.JPEG
      val/<synset>/*.JPEG

Label indices follow ``classes.py`` (HF ``IMAGENET2012_CLASSES`` OrderedDict).
Optional ``--synsets-file`` restricts extraction to a subset (e.g. CMC ImageNet-100).

Skip already-written files (idempotent resume). Does not write the unlabeled test split.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _load_synsets(classes_py: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("imagenet_classes", classes_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {classes_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.IMAGENET2012_CLASSES.keys())


def _load_allowed(synsets_file: Path | None, all_synsets: list[str]) -> set[str] | None:
    if synsets_file is None:
        return None
    allowed = {line.strip() for line in synsets_file.read_text().splitlines() if line.strip()}
    unknown = sorted(allowed - set(all_synsets))
    if unknown:
        raise ValueError(f"{len(unknown)} synsets not in ImageNet-1k, e.g. {unknown[:5]}")
    return allowed


def _process_shard(
    parquet_path: str,
    out_split_dir: str,
    synsets: list[str],
    allowed: frozenset[str] | None,
) -> tuple[int, int, str]:
    """Return (written, skipped, parquet_path)."""
    import pyarrow.parquet as pq

    out_root = Path(out_split_dir)
    table = pq.read_table(parquet_path, columns=["image", "label"])
    images = table["image"]
    labels = table["label"]
    written = 0
    skipped = 0
    n = table.num_rows
    for i in range(n):
        label = int(labels[i].as_py())
        synset = synsets[label]
        if allowed is not None and synset not in allowed:
            continue
        img = images[i].as_py()
        raw = img["bytes"]
        src_name = Path(img.get("path") or f"{synset}_{i}.JPEG").name
        # Keep original basename; ensure .JPEG suffix.
        if not src_name.lower().endswith((".jpeg", ".jpg")):
            src_name = f"{src_name}.JPEG"
        dest_dir = out_root / synset
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src_name
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(dest)
        written += 1
    return written, skipped, parquet_path


def _shards_for_split(data_dir: Path, split: str) -> list[Path]:
    if split == "train":
        return sorted(data_dir.glob("train-*.parquet"))
    if split == "val":
        # HF uses "validation-*.parquet"
        return sorted(data_dir.glob("validation-*.parquet"))
    raise ValueError(f"unknown split={split!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf-raw",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_dir" / "imagenet" / "hf_raw",
        help="Directory containing classes.py and data/*.parquet",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_dir" / "imagenet",
        help="Output root that will contain train/ and val/",
    )
    p.add_argument(
        "--synsets-file",
        type=Path,
        default=None,
        help="Optional text file of synsets to keep (e.g. CMC ImageNet-100).",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=("train", "val"),
    )
    args = p.parse_args()

    classes_py = args.hf_raw / "classes.py"
    data_dir = args.hf_raw / "data"
    if not classes_py.is_file():
        print(f"missing {classes_py}", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"missing {data_dir}", file=sys.stderr)
        return 1

    synsets = _load_synsets(classes_py)
    if len(synsets) != 1000:
        raise RuntimeError(f"expected 1000 synsets, got {len(synsets)}")
    allowed_set = _load_allowed(args.synsets_file, synsets)
    allowed = frozenset(allowed_set) if allowed_set is not None else None

    args.out_root.mkdir(parents=True, exist_ok=True)
    # torchvision / Avalanche expect "val", not "validation"
    split_out = {"train": args.out_root / "train", "val": args.out_root / "val"}

    print(f"hf_raw={args.hf_raw}")
    print(f"out_root={args.out_root}")
    print(f"synset_filter={'all' if allowed is None else f'{len(allowed)} classes'}")
    print(f"workers={args.workers} splits={args.splits}")

    total_w = total_s = 0
    for split in args.splits:
        shards = _shards_for_split(data_dir, split)
        out_dir = split_out[split]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {split}: {len(shards)} shards → {out_dir} ===", flush=True)
        if not shards:
            print(f"WARNING: no shards for {split}", file=sys.stderr)
            continue

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(_process_shard, str(sh), str(out_dir), synsets, allowed)
                for sh in shards
            ]
            done = 0
            for fut in as_completed(futs):
                w, s, path = fut.result()
                total_w += w
                total_s += s
                done += 1
                if done % 10 == 0 or done == len(futs):
                    print(
                        f"  [{done}/{len(futs)}] +{w} wrote / {s} skip  "
                        f"({Path(path).name})  cumulative_wrote={total_w}",
                        flush=True,
                    )

    # Quick summary
    for split, d in split_out.items():
        if split not in args.splits:
            continue
        n_cls = len([x for x in d.iterdir() if x.is_dir()]) if d.is_dir() else 0
        print(f"{split}: {n_cls} class folders under {d}")
    print(f"Done. wrote={total_w} skipped_existing={total_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
