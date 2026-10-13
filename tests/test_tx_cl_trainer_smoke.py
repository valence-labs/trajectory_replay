"""End-to-end CPU smoke for the sequential Tx trainer on a tiny h5ad."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]


def _write_tiny_dump(root: Path) -> tuple[Path, Path]:
    contexts: list[str] = []
    perts: list[str] = []
    splits: list[str] = []
    for ctx in ("jurkat", "k562", "rpe1"):
        for pert in ("non-targeting", "A"):
            contexts.extend([ctx, ctx])
            perts.extend([pert, pert])
            splits.extend(["train", "train"])
    contexts.extend(["hepg2"] * 6)
    perts.extend(["B", "B", "non-targeting", "non-targeting", "C", "C"])
    splits.extend(["finetune", "finetune", "finetune", "finetune", "test", "test"])

    rng = np.random.default_rng(0)
    se = rng.normal(size=(len(contexts), 4)).astype(np.float32)
    w = rng.normal(size=(3, 4)).astype(np.float32)
    genes = se @ w.T + rng.normal(scale=0.01, size=(len(contexts), 3))
    obs = pd.DataFrame(
        {
            "cell_line": pd.Categorical(contexts),
            "gene": pd.Categorical(perts),
            "split": pd.Categorical(splits, categories=["train", "finetune", "test"]),
        },
        index=[f"cell_{i}" for i in range(len(contexts))],
    )
    h5ad = root / "tiny.h5ad"
    se_path = root / "tiny_se.npy"
    ad.AnnData(X=genes.astype(np.float32), obs=obs).write_h5ad(h5ad)
    np.save(se_path, se)
    return h5ad, se_path


def test_sequential_trainer_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        h5ad, se = _write_tiny_dump(root)
        cases = (
            ("standard", "none", "ode", "full"),
            ("gft", "none", "ode", "embedding"),
            ("real", "equal", "ode", "full"),
            ("standard", "none", "ode", "lora"),
            ("generative", "equal", "ode", "full"),
            ("trajectory", "equal", "ode", "full"),
            ("generative", "grid", "ode", "full"),
            ("generative", "equal", "mean_flow", "full"),
            ("generative", "equal", "mean_flow", "embedding"),
            ("trajectory", "equal", "mean_flow", "full"),
        )
        for strategy, lambda_mode, replay_backend, scope in cases:
            out = root / f"{strategy}_{replay_backend}_{lambda_mode}_{scope}"
            command = [
                str(REPO / ".venv/bin/python"),
                str(REPO / "experiments/tx/train_cl.py"),
                "--dataset",
                "replogle",
                "--strategy",
                strategy,
                "--lambda-mode",
                lambda_mode,
                "--replay-backend",
                replay_backend,
                "--finetune-scope",
                scope,
                "--h5ad-path",
                str(h5ad),
                "--se-path",
                str(se),
                "--out-dir",
                str(out),
                "--device",
                "cpu",
                "--smoke",
                "--cond-emb-dim",
                "4",
                "--time-emb-dim",
                "8",
                "--hidden-dim",
                "16",
                "--depth",
                "2",
                "--batch-size",
                "8",
                "--eval-batch-size",
                "8",
                "--ode-step-size",
                "0.5",
                "--no-resume",
            ]
            subprocess.run(command, cwd=REPO, check=True, timeout=180)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            assert len(summary["metrics_by_task"]) == 2
            assert (out / "tasks/00/checkpoint.pt").is_file()
            assert (out / "tasks/01/checkpoint.pt").is_file()
            state_0 = torch.load(
                out / "tasks/00/checkpoint.pt", map_location="cpu", weights_only=False
            )["state_dict"]
            state_1 = torch.load(
                out / "tasks/01/checkpoint.pt", map_location="cpu", weights_only=False
            )["state_dict"]
            vf_keys = [key for key in state_0 if key.startswith("mlp.")]
            if scope == "embedding":
                assert all(torch.equal(state_0[key], state_1[key]) for key in vf_keys)
            if strategy == "standard" and scope == "lora":
                assert any(not torch.equal(state_0[key], state_1[key]) for key in vf_keys)
                assert not any("lora_" in key for key in state_1)
            if replay_backend == "mean_flow":
                assert (out / "tasks/00/mean_flow.pt").is_file()
                assert (out / "tasks/01/mean_flow.pt").is_file()
            assert (out / "decoders/decoder_task00.npz").is_file()
            assert (out / "decoders/decoder_task01.npz").is_file()
            sweep = json.loads(
                (out / "tasks/01/lambda_sweep.json").read_text(encoding="utf-8")
            )
            if lambda_mode == "none":
                assert sweep["chosen_lambda"] is None
                assert sweep["selection_evaluated"] is False
                assert sweep["candidates"][0]["evaluation"] == "skipped"
            elif lambda_mode == "equal":
                assert sweep["chosen_lambda"] == 0.5
                assert sweep["selection_evaluated"] is False
                assert sweep["candidates"][0]["evaluation"] == "skipped"
            else:
                assert sweep["chosen_lambda"] in (0.25, 0.5, 0.75)
                assert sweep["selection_evaluated"] is True
                assert len(sweep["candidates"]) == 3
            assert (out / "final_metrics.json").is_file()
            final = summary["final"]["final"]
            assert "fit" in final and "heldout" in final
            if lambda_mode in ("none", "equal"):
                curve_cmd = [
                    str(REPO / ".venv/bin/python"),
                    str(REPO / "experiments/tx/eval_cl_heldout.py"),
                    "--out-dir",
                    str(out),
                    "--device",
                    "cpu",
                ]
                subprocess.run(curve_cmd, cwd=REPO, check=True, timeout=60)
                curve = json.loads((out / "heldout_curve.json").read_text(encoding="utf-8"))
                assert len(curve["heldout_curve"]) == 2
                assert curve["heldout_curve"][0]["heldout"]["n_cells"] == 0
                assert curve["heldout_curve"][1]["heldout"]["n_cells"] == 2
                assert (
                    curve["heldout_curve"][1]["heldout"]["n_control_reference_cells"]
                    == 2
                )
                assert (
                    curve["heldout_curve"][1]["heldout"]["source"]
                    == "final_metrics.json"
                )


def test_scimilarity_representation_paths() -> None:
    from argparse import Namespace

    from experiments.tx.train_cl import (
        DEFAULTS,
        cl_representation,
        embedding_sidecar_path,
        shared_decoder_dirname,
    )

    assert cl_representation(Namespace(representation="scimilarity")) == "scimilarity"
    assert shared_decoder_dirname("scimilarity") == "decoders_scimilarity"
    assert shared_decoder_dirname("scimilarity", hidden=1024, depth=2) == (
        "decoders_scimilarity_mlp_h1024_d2"
    )
    path = embedding_sidecar_path(DEFAULTS["replogle"], "scimilarity", None)
    assert path == DEFAULTS["replogle"]["scimilarity"]
    custom = Path("/tmp/custom_sci.npy")
    assert embedding_sidecar_path(DEFAULTS["replogle"], "scimilarity", custom) == custom


def test_sequential_trainer_gene_space_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        h5ad, se = _write_tiny_dump(root)
        out = root / "gene_standard"
        command = [
            str(REPO / ".venv/bin/python"),
            str(REPO / "experiments/tx/train_cl.py"),
            "--dataset",
            "replogle",
            "--strategy",
            "standard",
            "--lambda-mode",
            "none",
            "--representation",
            "genes",
            "--h5ad-path",
            str(h5ad),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--smoke",
            "--cond-emb-dim",
            "4",
            "--time-emb-dim",
            "8",
            "--hidden-dim",
            "16",
            "--depth",
            "2",
            "--batch-size",
            "8",
            "--eval-batch-size",
            "8",
            "--ode-step-size",
            "0.5",
            "--no-resume",
        ]
        subprocess.run(command, cwd=REPO, check=True, timeout=180)
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert cfg["representation"] == "genes"
        dec = np.load(out / "decoders/decoder_task00.npz")
        assert bool(dec["identity"])
        ckpt = torch.load(out / "tasks/00/checkpoint.pt", map_location="cpu", weights_only=False)
        assert int(ckpt["config"]["data_dim"]) == 3
        assert (out / "final_metrics.json").is_file()


if __name__ == "__main__":
    test_sequential_trainer_smoke()
    test_sequential_trainer_gene_space_smoke()
    test_scimilarity_representation_paths()
    print("ok")

