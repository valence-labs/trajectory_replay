"""Fast tests for stage-C SE datasets and λ selection metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trajectory_replay.data.tx_cl_data import (
    IndexedSeDataset,
    TxClArrays,
    stratified_cap_indices,
)
from trajectory_replay.data.tx_cl_tasks import replogle_stream
from trajectory_replay.data.tx_dataset import build_encoding, encode_labels
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank, TaskLinearDecoder
from trajectory_replay.eval.tx_cl_pearson import (
    decode_observed_se,
    pearson_vector,
    sample_matched_control_indices,
    score_gene_strata,
    stratum_generation_plan,
)
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from experiments.tx.train_cl import (
    EVAL_METRIC_VERSION,
    EVAL_TRACK_SEED_OFFSETS,
    cap_eval_indices,
    control_reference_indices,
    epochs_for_target_steps,
    eval_provenance,
    eval_seed,
    steps_per_epoch,
    task_finetune_params,
)


def _arrays(tmp: Path) -> TxClArrays:
    context = np.array(["k562"] * 4 + ["hepg2"] * 4, dtype=object)
    perturbation = np.array(
        [
            "non-targeting",
            "non-targeting",
            "A",
            "A",
            "non-targeting",
            "non-targeting",
            "C",
            "C",
        ],
        dtype=object,
    )
    split = np.array(["train"] * 4 + ["finetune"] * 4, dtype=object)
    enc = build_encoding(
        context,
        perturbation,
        control_perturbation="non-targeting",
    )
    c, p, z = encode_labels(context, perturbation, enc)
    se_path = tmp / "se.npy"
    se = np.arange(32, dtype=np.float32).reshape(8, 4)
    np.save(se_path, se)
    return TxClArrays(
        h5ad_path=tmp / "fake.h5ad",
        se_path=se_path,
        stream=replogle_stream(),
        encoding=enc,
        context=context,
        perturbation=perturbation,
        split=split,
        c=c,
        p=p,
        z=z,
        se=se,
        se_dim=4,
        n_cells=8,
    )


def test_indexed_se_dataset() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        ds = IndexedSeDataset(arrays, np.array([6, 1]))
        x, c, p = ds[0]
        np.testing.assert_array_equal(x.numpy(), np.arange(24, 28, dtype=np.float32))
        assert int(c) == arrays.c[6]
        assert int(p) == arrays.p[6]


def test_stratified_cap() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = stratified_cap_indices(
            np.arange(8),
            arrays.c,
            arrays.p,
            arrays.z,
            (),
            max_per_stratum=1,
            seed=0,
        )
        assert idx.size == 4
        keys = set(zip(arrays.c[idx].tolist(), arrays.p[idx].tolist()))
        assert len(keys) == 4


def test_heldout_control_reference_uses_same_context() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        controls = control_reference_indices(
            arrays,
            target_indices=np.array([7]),  # HepG2 C
            trainable_indices=np.arange(7),
            max_per_stratum=1,
            seed=0,
        )
        assert controls.size == 1
        assert arrays.context[controls[0]] == "hepg2"
        assert arrays.perturbation[controls[0]] == "non-targeting"


def test_generation_plan_identity_when_disabled() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = np.arange(8)
        task_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        pred_idx, pred_tids, pred_src = stratum_generation_plan(
            arrays,
            idx,
            task_ids,
            samples_per_stratum=0,
            control_pert_index=arrays.encoding.control_pert_index,
        )
        np.testing.assert_array_equal(pred_idx, idx)
        np.testing.assert_array_equal(pred_tids, task_ids)
        np.testing.assert_array_equal(pred_src, np.arange(8))


def test_generation_plan_fixed_k_is_condition_exact() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = np.arange(8)
        task_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        control = arrays.encoding.control_pert_index
        k = 5
        pred_idx, pred_tids, pred_src = stratum_generation_plan(
            arrays,
            idx,
            task_ids,
            samples_per_stratum=k,
            control_pert_index=control,
        )
        is_ctrl = arrays.p[pred_idx] == int(control)
        # Two perturbed strata (k562 A, hepg2 C) get exactly k rows each,
        # and every observed control is carried through untouched.
        assert int((~is_ctrl).sum()) == 2 * k
        np.testing.assert_array_equal(
            pred_idx[is_ctrl], idx[arrays.p[idx] == int(control)]
        )
        # Generated rows have no observed counterpart; controls keep theirs.
        assert set(pred_src[~is_ctrl].tolist()) == {-1}
        assert np.all(pred_src[is_ctrl] >= 0)
        # Conditioning is constant inside each replicated stratum.
        for pert in np.unique(arrays.p[pred_idx[~is_ctrl]]):
            rows = pred_idx[~is_ctrl][arrays.p[pred_idx[~is_ctrl]] == pert]
            assert rows.size == k
            assert np.unique(arrays.c[rows]).size == 1
            assert np.unique(pred_tids[~is_ctrl][arrays.p[pred_idx[~is_ctrl]] == pert]).size == 1


def test_score_gene_strata_accepts_replicated_predictions() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = np.array([1, 3, 5, 7])
        true = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 3.0, 4.0],
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
            ]
        )
        task_ids = np.array([0, 0, 1, 1])
        baseline = score_gene_strata(
            arrays,
            idx,
            true,
            true.copy(),
            task_ids,
            control_pert_index=arrays.encoding.control_pert_index,
        )
        # Duplicating each predicted row leaves every stratum mean unchanged.
        pred_idx = np.repeat(idx, 3)
        pred = np.repeat(true, 3, axis=0)
        replicated = score_gene_strata(
            arrays,
            idx,
            true,
            pred,
            task_ids,
            control_pert_index=arrays.encoding.control_pert_index,
            pred_indices=pred_idx,
        )
        assert replicated["pearson_delta_macro"] == baseline["pearson_delta_macro"]
        assert replicated["n_delta_strata"] == baseline["n_delta_strata"]


def test_score_gene_strata_rejects_unpredicted_stratum() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = np.array([1, 3, 5, 7])
        true = np.zeros((4, 3))
        true[1] = [2.0, 3.0, 4.0]
        task_ids = np.array([0, 0, 1, 1])
        try:
            score_gene_strata(
                arrays,
                idx,
                true,
                true[:3],
                task_ids,
                control_pert_index=arrays.encoding.control_pert_index,
                pred_indices=idx[:3],
            )
        except ValueError as err:
            assert "no prediction" in str(err)
        else:
            raise AssertionError("expected a missing-prediction error")


def test_selection_uses_delta_for_every_trainable_context() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = np.array([1, 3, 5, 7])  # one cell per stratum
        true = np.array(
            [
                [1.0, 1.0, 1.0],  # k562 basal
                [2.0, 3.0, 4.0],  # k562 A: has basal
                [3.0, 2.0, 1.0],  # hepg2 basal
                [1.0, 3.0, 2.0],  # hepg2 C: has basal
            ]
        )
        pred = true.copy()
        task_ids = np.array([0, 0, 1, 1])
        out = score_gene_strata(
            arrays,
            idx,
            true,
            pred,
            task_ids,
            control_pert_index=arrays.encoding.control_pert_index,
        )
        assert out["selection_metric"] == "pearson_delta"
        assert out["n_delta_strata"] == 2
        assert out["n_residual_strata"] == 0
        assert out["n_pearson_fallback_strata"] == 0
        assert abs(out["selection_score"] - 1.0) < 1e-12
        assert np.isnan(out["pearson_delta_residual_macro"])
        assert np.isnan(out["per_task"]["0"]["pearson_delta_residual"])
        assert out["per_task"]["0"]["n_residual_strata"] == 0
        assert abs(out["per_task"]["0"]["pearson_delta"] - 1.0) < 1e-12
        assert abs(out["per_task"]["1"]["pearson_delta"] - 1.0) < 1e-12


def test_residual_nan_for_singleton_pool_zero_for_geneblind() -> None:
    with tempfile.TemporaryDirectory() as td:
        context = np.array(["k562"] * 6, dtype=object)
        perturbation = np.array(
            ["non-targeting", "non-targeting", "A", "A", "B", "B"],
            dtype=object,
        )
        split = np.array(["train"] * 6, dtype=object)
        enc = build_encoding(
            context, perturbation, control_perturbation="non-targeting"
        )
        c, p, z = encode_labels(context, perturbation, enc)
        se = np.arange(24, dtype=np.float32).reshape(6, 4)
        se_path = Path(td) / "se.npy"
        np.save(se_path, se)
        arrays = TxClArrays(
            h5ad_path=Path(td) / "fake.h5ad",
            se_path=se_path,
            stream=replogle_stream(),
            encoding=enc,
            context=context,
            perturbation=perturbation,
            split=split,
            c=c,
            p=p,
            z=z,
            se=se,
            se_dim=4,
            n_cells=6,
        )
        idx = np.arange(6)
        task_ids = np.zeros(6, dtype=np.int64)
        true = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        identity = score_gene_strata(
            arrays,
            idx,
            true,
            true.copy(),
            task_ids,
            control_pert_index=arrays.encoding.control_pert_index,
        )
        assert identity["n_delta_strata"] == 2
        assert identity["n_residual_strata"] == 2
        assert identity["pearson_delta_residual_macro"] > 0.99

        geneblind = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        blind = score_gene_strata(
            arrays,
            idx,
            true,
            geneblind,
            task_ids,
            control_pert_index=arrays.encoding.control_pert_index,
        )
        assert blind["n_residual_strata"] == 2
        assert abs(blind["pearson_delta_residual_macro"]) < 1e-12


def test_old_new_balanced_selection_equalizes_task_macros() -> None:
    from trajectory_replay.eval.tx_cl_pearson import old_new_balanced_selection

    result = {
        "rows": (
            [{"task_id": 0, "selection_score": 0.1}] * 9
            + [{"task_id": 1, "selection_score": 0.9}]
        )
    }
    out = old_new_balanced_selection(result, new_task_id=1)
    assert out["selection_metric"] == "old_new_balanced_pearson_delta"
    assert out["n_old_strata"] == 9
    assert out["n_new_strata"] == 1
    assert abs(out["old_task_macro"] - 0.1) < 1e-12
    assert abs(out["new_task_macro"] - 0.9) < 1e-12
    assert abs(out["selection_score"] - 0.5) < 1e-12
    pooled = (9 * 0.1 + 0.9) / 10
    assert abs(pooled - 0.18) < 1e-12


def test_full_replay_unfreezes_every_model_parameter() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        model = ConditionalFlowMatching(
            data_dim=arrays.se_dim,
            n_contexts=arrays.encoding.n_contexts,
            n_perturbations=arrays.encoding.n_perturbations,
            covariate_sizes={"batch": 2},
        )
        args = SimpleNamespace(
            strategy="real",
            finetune_scope="full",
            finetune_train_old_pert_embeddings=False,
            finetune_train_old_context_embeddings=False,
        )
        params, context_rows, pert_rows = task_finetune_params(
            model, arrays, np.array([4, 5, 6, 7]), args, stage=None
        )
        assert context_rows is None
        assert pert_rows is None
        assert {id(p) for p in params} == {id(p) for p in model.parameters()}
        assert all(p.requires_grad for p in model.parameters())


def test_naive_trains_novel_pert_rows_and_freezes_old_ones() -> None:
    """pert=0 freezes T0 perturbation rows; HepG2-only genes still train."""
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        model = ConditionalFlowMatching(
            data_dim=arrays.se_dim,
            n_contexts=arrays.encoding.n_contexts,
            n_perturbations=arrays.encoding.n_perturbations,
            covariate_sizes={"batch": 2},
        )
        args = SimpleNamespace(
            strategy="standard",
            finetune_scope="full",
            finetune_train_old_pert_embeddings=False,
            finetune_train_old_context_embeddings=False,
        )
        new_idx = np.array([4, 5, 6, 7])
        old_idx = np.array([0, 1, 2, 3])
        params, context_rows, pert_rows = task_finetune_params(
            model, arrays, new_idx, args, stage=None, old_indices=old_idx
        )
        novel = {int(x) for x in np.unique(arrays.p[new_idx])} - {
            int(x) for x in np.unique(arrays.p[old_idx])
        }
        assert pert_rows == frozenset(novel)
        assert novel
        assert context_rows == frozenset(int(x) for x in np.unique(arrays.c[new_idx]))
        assert model.pert_emb.weight.requires_grad
        names = {id(p) for p in params}
        assert id(model.pert_emb.weight) in names
        assert id(model.context_emb.weight) in names


def test_pearson_delta_uses_decoded_pred_basal() -> None:
    """Real basal from observed controls; pred basal from decoded controls."""
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = np.array([1, 3, 5, 7])
        true = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 3.0, 4.0],
                [3.0, 2.0, 1.0],
                [4.0, 4.0, 4.0],
            ]
        )
        pred = np.array(
            [
                [0.0, 2.0, 1.0],  # decoded k562 control
                [1.0, 4.0, 4.0],  # decoded k562 A: same delta as true
                [1.0, 1.0, 0.0],
                [2.0, 3.0, 3.0],
            ]
        )
        task_ids = np.array([0, 0, 1, 1])
        out = score_gene_strata(
            arrays,
            idx,
            true,
            pred,
            task_ids,
            control_pert_index=arrays.encoding.control_pert_index,
        )
        assert abs(out["pearson_delta_macro"] - 1.0) < 1e-12
        shared_basal = pearson_vector(true[1] - true[0], pred[1] - true[0])
        assert shared_basal < 1.0 - 1e-6


def test_apply_pred_control_observed_copies_true_basal() -> None:
    from trajectory_replay.eval.tx_cl_pearson import apply_pred_control_genes

    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        idx = np.array([1, 3])
        true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pred = np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])
        mixed = apply_pred_control_genes(
            arrays, idx, true, pred, mode="observed"
        )
        np.testing.assert_array_equal(mixed[0], true[0])
        np.testing.assert_array_equal(mixed[1], pred[1])
        same = apply_pred_control_genes(arrays, idx, true, pred, mode="decoded")
        np.testing.assert_array_equal(same, pred)
        generated = apply_pred_control_genes(
            arrays, idx, true, pred, mode="generated"
        )
        np.testing.assert_array_equal(generated, pred)


def test_decode_observed_se_is_linear_reconstruction() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        bank = TaskDecoderBank()
        identity = np.eye(arrays.se_dim, dtype=np.float64)
        bias = np.zeros(arrays.se_dim, dtype=np.float64)
        for tid in (0, 1):
            bank.add(
                TaskLinearDecoder(
                    task_id=tid,
                    weight=identity,
                    intercept=bias,
                    n_fit=1,
                    lr=1e-3,
                    weight_decay=0.0,
                    epochs=1,
                )
            )
        idx = np.array([0, 4])
        recon = decode_observed_se(arrays, bank, idx, np.array([0, 1]))
        np.testing.assert_allclose(recon, arrays.se[idx])


def test_d2d_sources_are_controls_from_the_matching_context() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        target = np.array([2, 3, 6, 7])
        source = np.arange(arrays.n_cells)
        sampled = sample_matched_control_indices(
            arrays, target, source, seed=123
        )
        control = int(arrays.encoding.control_pert_index)
        np.testing.assert_array_equal(arrays.p[sampled], control)
        np.testing.assert_array_equal(arrays.c[sampled], arrays.c[target])
        np.testing.assert_array_equal(
            sampled,
            sample_matched_control_indices(arrays, target, source, seed=123),
        )


def test_d2d_sources_reject_missing_context_controls() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        try:
            sample_matched_control_indices(
                arrays, np.array([6]), np.array([0, 1]), seed=0
            )
        except ValueError as exc:
            assert "No matched control source" in str(exc)
        else:
            raise AssertionError("missing matched controls must fail")


def _controls_of(arrays: TxClArrays, idx: np.ndarray) -> np.ndarray:
    return idx[arrays.p[idx] == int(arrays.encoding.control_pert_index)]


def test_cap_eval_indices_keeps_the_whole_basal() -> None:
    """Scored strata are capped; control strata are not.

    Every stratum in a pool subtracts the same control mean, so a subsampled
    basal shifts all of that pool's deltas together rather than averaging out.
    """
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        args = SimpleNamespace(
            eval_max_cells_per_stratum=1,
            eval_max_control_cells=0,
        )
        idx = cap_eval_indices(arrays, np.arange(8), args, seed=0)
        controls = _controls_of(arrays, idx)
        assert controls.size == 4  # every control cell survives
        assert idx.size - controls.size == 2  # one per perturbed stratum
        redrawn = cap_eval_indices(arrays, np.arange(8), args, seed=12345)
        np.testing.assert_array_equal(controls, _controls_of(arrays, redrawn))


def test_cap_eval_indices_honours_an_explicit_control_cap() -> None:
    with tempfile.TemporaryDirectory() as td:
        arrays = _arrays(Path(td))
        args = SimpleNamespace(
            eval_max_cells_per_stratum=1,
            eval_max_control_cells=1,
        )
        idx = cap_eval_indices(arrays, np.arange(8), args, seed=0)
        assert _controls_of(arrays, idx).size == 2  # one per context


def test_eval_seed_is_unique_per_track_and_task() -> None:
    seeds = {
        eval_seed(42, track, task)
        for track in EVAL_TRACK_SEED_OFFSETS
        for task in range(9)
    }
    assert len(seeds) == len(EVAL_TRACK_SEED_OFFSETS) * 9
    try:
        eval_seed(42, "not_a_track", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown track must raise")


def test_heldout_curve_reuses_final_metrics_when_provenance_matches() -> None:
    """The curve's last point and ``final_metrics`` must be one number."""
    from experiments.tx.eval_cl_heldout import reusable_final_heldout

    cfg = SimpleNamespace(
        seed=42,
        eval_max_cells_per_stratum=500,
        eval_max_control_cells=0,
        ode_method="euler",
        ode_step_size=0.05,
        ode_rtol=1e-5,
        ode_atol=1e-5,
        pred_control="decoded",
    )
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        (out_dir / "final_metrics.json").write_text(
            json.dumps(
                {
                    "eval_provenance": eval_provenance(
                        cfg, 1, tracks=("fit", "forget", "stream", "heldout")
                    ),
                    "final": {"heldout": {"pearson_delta_macro": 0.5}},
                }
            ),
            encoding="utf-8",
        )
        reused = reusable_final_heldout(out_dir, cfg, 1)
        assert reused is not None and reused["pearson_delta_macro"] == 0.5
        assert reusable_final_heldout(out_dir, cfg, 0) is None
        drifted = SimpleNamespace(**{**vars(cfg), "ode_step_size": 0.01})
        assert reusable_final_heldout(out_dir, drifted, 1) is None
        stale = SimpleNamespace(**{**vars(cfg), "eval_max_control_cells": 500})
        assert reusable_final_heldout(out_dir, stale, 1) is None

        old_prov = eval_provenance(cfg, 1, tracks=("fit", "forget", "stream", "heldout"))
        assert old_prov["metric_version"] == EVAL_METRIC_VERSION
        old_prov.pop("metric_version")
        (out_dir / "final_metrics.json").write_text(
            json.dumps(
                {
                    "eval_provenance": old_prov,
                    "final": {"heldout": {"pearson_delta_macro": 0.5}},
                }
            ),
            encoding="utf-8",
        )
        assert reusable_final_heldout(out_dir, cfg, 1) is None


def test_epochs_for_target_steps_matches_replogle_budget() -> None:
    assert steps_per_epoch(546797, 1024) == 534
    assert epochs_for_target_steps(546797, 1024, 100_000) == 188
    assert steps_per_epoch(30874, 1024) == 31
    assert epochs_for_target_steps(30874, 1024, 100_000) == 3226
    assert 534 * 188 >= 100_000
    assert 31 * 3226 >= 100_000


if __name__ == "__main__":
    test_indexed_se_dataset()
    test_stratified_cap()
    test_heldout_control_reference_uses_same_context()
    test_selection_uses_delta_for_every_trainable_context()
    test_residual_nan_for_singleton_pool_zero_for_geneblind()
    test_old_new_balanced_selection_equalizes_task_macros()
    test_full_replay_unfreezes_every_model_parameter()
    test_pearson_delta_uses_decoded_pred_basal()
    test_apply_pred_control_observed_copies_true_basal()
    test_decode_observed_se_is_linear_reconstruction()
    test_d2d_sources_are_controls_from_the_matching_context()
    test_d2d_sources_reject_missing_context_controls()
    test_cap_eval_indices_keeps_the_whole_basal()
    test_cap_eval_indices_honours_an_explicit_control_cap()
    test_eval_seed_is_unique_per_track_and_task()
    test_heldout_curve_reuses_final_metrics_when_provenance_matches()
    test_epochs_for_target_steps_matches_replogle_budget()
    print("ok")

