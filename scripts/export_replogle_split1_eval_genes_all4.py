#!/usr/bin/env python3
"""Write genes present in all Replogle cell lines to the split1 eval allowlist."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--h5ad-path",
        type=Path,
        default=Path("data_dir/replogle_with_splits.h5ad"),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("configs/replogle_split1_eval_genes_all4.txt"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    obs = adata.obs[["cell_line", "gene"]].astype(str)
    by_line = {
        cl: set(obs.loc[obs["cell_line"] == cl, "gene"].unique())
        for cl in obs["cell_line"].unique()
    }
    genes = sorted(set.intersection(*by_line.values()))
    header = (
        "# Genes with cells in all four Replogle cell lines (hepg2, jurkat, k562, rpe1).\n"
        "# Used for ODE/VCB eval only; training uses the full perturbation vocabulary.\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + "\n".join(genes) + "\n", encoding="utf-8")
    print(f"wrote {len(genes)} genes to {args.output}")


if __name__ == "__main__":
    main()
