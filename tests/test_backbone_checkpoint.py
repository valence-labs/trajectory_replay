"""Tests for swappable backbones and legacy checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path

import torch

from trajectory_replay.model.backbones import build_mlp_sequential
from trajectory_replay.model.flow_map import (
    ConditionalFlowMap,
    build_conditional_flow_map_from_config,
)
from trajectory_replay.model.flow_matching import (
    ConditionalFlowMatching,
    build_conditional_flow_matching_from_config,
)
from trajectory_replay.model.lora import LoRALinear
from trajectory_replay.training.mean_flow_distill import make_mean_flow_from_teacher
from trajectory_replay.training.flow_map_distill import (
    load_flow_map_checkpoint,
    save_flow_map_checkpoint,
)


def _legacy_flow_matching_cfg() -> dict:
    return {
        "data_dim": 4,
        "n_contexts": 2,
        "n_perturbations": 2,
        "cond_emb_dim": 8,
        "time_emb_dim": 8,
        "hidden_dim": 16,
        "depth": 3,
    }


def test_mlp_backbone_keeps_legacy_state_dict_keys() -> None:
    model = ConditionalFlowMatching(**_legacy_flow_matching_cfg())
    keys = list(model.state_dict())
    assert any(k.startswith("mlp.") for k in keys)
    assert not any(k.startswith("backbone.") for k in keys)
    assert model.backbone_type == "mlp"


def test_injected_mlp_backbone_roundtrip() -> None:
    cfg = _legacy_flow_matching_cfg()
    in_dim = cfg["data_dim"] + cfg["time_emb_dim"] + 2 * cfg["cond_emb_dim"]
    backbone = build_mlp_sequential(
        in_dim,
        cfg["data_dim"],
        hidden_dim=cfg["hidden_dim"],
        depth=cfg["depth"],
    )
    model = ConditionalFlowMatching(**cfg, backbone=backbone)
    x = torch.randn(3, cfg["data_dim"])
    t = torch.rand(3)
    c = torch.tensor([0, 1, 0])
    p = torch.tensor([0, 0, 1])
    out1 = model(x, t, c, p)

    payload = {"state_dict": model.state_dict(), "config": {**cfg, "backbone_type": "mlp"}}
    rebuilt = build_conditional_flow_matching_from_config(payload["config"])
    rebuilt.load_state_dict(payload["state_dict"])
    out2 = rebuilt(x, t, c, p)
    assert torch.allclose(out1, out2)


def test_legacy_checkpoint_without_backbone_type_field(tmp_path: Path) -> None:
    cfg = _legacy_flow_matching_cfg()
    model = ConditionalFlowMatching(**cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt)

    payload = torch.load(ckpt, weights_only=False)
    loaded = build_conditional_flow_matching_from_config(payload["config"])
    loaded.load_state_dict(payload["state_dict"])

    x = torch.randn(2, cfg["data_dim"])
    t = torch.rand(2)
    c = torch.zeros(2, dtype=torch.long)
    p = torch.zeros(2, dtype=torch.long)
    assert torch.allclose(model(x, t, c, p), loaded(x, t, c, p))
    assert loaded.backbone_type == "mlp"


def test_flow_map_legacy_checkpoint_without_backbone_type(tmp_path: Path) -> None:
    cfg = {
        **_legacy_flow_matching_cfg(),
        "phi_rescale": 1.0,
        "model_type": "conditional_flow_map",
    }
    flow_map = ConditionalFlowMap(
        data_dim=cfg["data_dim"],
        n_contexts=cfg["n_contexts"],
        n_perturbations=cfg["n_perturbations"],
        cond_emb_dim=cfg["cond_emb_dim"],
        time_emb_dim=cfg["time_emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        depth=cfg["depth"],
    )
    ckpt = tmp_path / "flow_map.pt"
    save_flow_map_checkpoint(
        ckpt,
        flow_map,
        {"stage": "pretrain"},
        parent_checkpoint="checkpoint.pt",
    )
    loaded, loaded_cfg = load_flow_map_checkpoint(ckpt, torch.device("cpu"))
    assert loaded_cfg.get("backbone_type", "mlp") == "mlp"

    x = torch.randn(2, cfg["data_dim"])
    s = torch.zeros(2)
    t = torch.ones(2)
    c = torch.zeros(2, dtype=torch.long)
    p = torch.zeros(2, dtype=torch.long)
    assert torch.allclose(flow_map(x, s, t, c, p), loaded(x, s, t, c, p))


def test_build_flow_map_from_config_matches_constructor() -> None:
    cfg = {**_legacy_flow_matching_cfg(), "phi_rescale": 0.5, "backbone_type": "mlp"}
    a = ConditionalFlowMap(phi_rescale=0.5, **_legacy_flow_matching_cfg())
    b = build_conditional_flow_map_from_config(cfg)
    assert a.phi_rescale == b.phi_rescale
    assert a.backbone_type == b.backbone_type


def test_residual_backbone_supports_lora_roundtrip() -> None:
    model = ConditionalFlowMatching(
        **_legacy_flow_matching_cfg(),
        backbone_type="residual_mlp",
    )
    x = torch.randn(3, 4)
    t = torch.rand(3)
    c = torch.tensor([0, 1, 0])
    p = torch.tensor([0, 0, 1])
    before = model(x, t, c, p)

    model.apply_mlp_lora(rank=2, alpha=4)
    assert any(isinstance(module, LoRALinear) for module in model.backbone.modules())
    params, _, _ = model.configure_finetune(
        holdout_mode="context",
        train_vector_field=False,
        train_lora=True,
        held_out_pert=0,
        held_out_context=1,
    )
    assert params
    assert torch.allclose(before, model(x, t, c, p))

    model.merge_mlp_lora()
    assert not any(isinstance(module, LoRALinear) for module in model.backbone.modules())
    assert torch.allclose(before, model(x, t, c, p))


def test_residual_teacher_builds_residual_mean_flow() -> None:
    teacher = ConditionalFlowMatching(
        **_legacy_flow_matching_cfg(),
        backbone_type="residual_mlp",
    )
    mean_flow = make_mean_flow_from_teacher(teacher)
    assert mean_flow.backbone_type == "residual_mlp"

    x = torch.randn(2, 4)
    r = torch.zeros(2)
    t = torch.ones(2)
    c = torch.tensor([0, 1])
    p = torch.tensor([1, 0])
    assert mean_flow(x, r, t, c, p).shape == x.shape
