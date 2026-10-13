#!/usr/bin/env python3
"""Print sequential-CL train / forget / held-out cell counts from dump obs.

Reads only categorical ``obs`` columns (no ``X``). Safe on a login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_replay.data.tx_cl_tasks import (
    load_stream_obs,
    stream_for,
    summarize_stream,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5AD = {
    "pbmc": REPO_ROOT / "data_dir/pbmc_parse_with_splits.h5ad",
    "replogle": REPO_ROOT / "data_dir/replogle_with_splits.h5ad",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pbmc", "replogle"), required=True)
    parser.add_argument("--h5ad-path", type=Path, default=None)
    args = parser.parse_args()

    stream = stream_for(args.dataset)
    path = args.h5ad_path or DEFAULT_H5AD[args.dataset]
    if not path.is_file():
        raise SystemExit(f"missing dump: {path}")

    obs = load_stream_obs(path, stream)
    rows = summarize_stream(stream, obs["context"], obs["split"], obs["perturbation"])
    print(json.dumps({"dataset": args.dataset, "h5ad": str(path), "steps": rows}, indent=2))


if __name__ == "__main__":
    main()
