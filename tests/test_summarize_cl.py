from experiments.tx.summarize_cl import method_label


def test_method_label_separates_pca_widths() -> None:
    base = {
        "strategy": "real",
        "finetune_scope": "full",
        "lambda_mode": "grid",
        "replay_backend": "ode",
        "mean_flow_steps": 1,
    }
    pca50 = method_label({**base, "representation": "pca", "pca_dim": 50})
    pca256 = method_label({**base, "representation": "pca", "pca_dim": 256})
    se = method_label({**base, "representation": "se"})
    assert pca50 != pca256
    assert pca50.startswith("pca50 ")
    assert pca256.startswith("pca256 ")
    assert se.startswith("se ")
    assert "real full grid" in pca50
