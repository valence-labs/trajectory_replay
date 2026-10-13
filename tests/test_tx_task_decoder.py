"""Unit tests for per-task linear SE→gene decoders (tiny matrices)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trajectory_replay.data.tx_task_decoder import (
    TaskDecoderBank,
    TaskLinearDecoder,
    TaskMLPDecoder,
    fit_task_decoder,
    fit_task_decoder_from_batches,
    identity_task_decoder,
    load_task_decoder,
)

_FIT = dict(epochs=80, lr=5e-2, batch_size=32, seed=0)


def _linear_data(n: int, se_dim: int, n_genes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n_genes, se_dim))
    b = rng.normal(size=(n_genes,))
    se = rng.normal(size=(n, se_dim))
    genes = se @ w.T + b + 0.01 * rng.normal(size=(n, n_genes))
    return se, genes


def test_fit_decode_roundtrip() -> None:
    se, genes = _linear_data(80, se_dim=8, n_genes=5, seed=0)
    dec = fit_task_decoder(se, genes, task_id=0, **_FIT)
    recon = dec.decode(se)
    corr = np.corrcoef(genes.ravel(), recon.ravel())[0, 1]
    assert corr > 0.99, corr
    assert dec.n_fit == 80
    assert dec.weight.shape == (5, 8)


def test_save_load(tmp_path: Path | None = None) -> None:
    se, genes = _linear_data(20, 4, 3, seed=2)
    dec = fit_task_decoder(se, genes, task_id=1, **_FIT)
    if tmp_path is None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            _save_load_check(Path(td), dec, se)
    else:
        _save_load_check(tmp_path, dec, se)


def _save_load_check(directory: Path, dec: TaskLinearDecoder, se: np.ndarray) -> None:
    path = directory / "decoder_task01.npz"
    dec.save(path)
    loaded = TaskLinearDecoder.load(path)
    np.testing.assert_allclose(loaded.decode(se), dec.decode(se))
    assert loaded.task_id == 1


def test_bank_freeze_and_per_task_decode() -> None:
    se0, g0 = _linear_data(30, 6, 4, seed=3)
    se1, g1 = _linear_data(30, 6, 4, seed=4)
    bank = TaskDecoderBank()
    bank.add(fit_task_decoder(se0, g0, task_id=0, **_FIT))
    try:
        bank.add(fit_task_decoder(se0, g0, task_id=0, **_FIT))
    except ValueError as exc:
        assert "frozen" in str(exc)
    else:
        raise AssertionError("expected freeze error")
    try:
        bank.add(fit_task_decoder(se1, g1, task_id=2, **_FIT))
    except ValueError as exc:
        assert "Next decoder" in str(exc)
    else:
        raise AssertionError("expected order error")
    bank.add(fit_task_decoder(se1, g1, task_id=1, **_FIT))
    se = np.vstack([se0[:5], se1[:5]])
    ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    out = bank.decode(se, ids)
    np.testing.assert_allclose(out[:5], bank.get(0).decode(se0[:5]))
    np.testing.assert_allclose(out[5:], bank.get(1).decode(se1[:5]))
    try:
        bank.decode(se[:1], np.array([-1]))
    except ValueError as exc:
        assert "-1" in str(exc)
    else:
        raise AssertionError("expected refuse test/untrained cells")


def test_bank_save_dir() -> None:
    import tempfile

    se0, g0 = _linear_data(16, 4, 3, seed=5)
    se1, g1 = _linear_data(16, 4, 3, seed=6)
    bank = TaskDecoderBank()
    bank.add(fit_task_decoder(se0, g0, task_id=0, **_FIT))
    bank.add(fit_task_decoder(se1, g1, task_id=1, **_FIT))
    with tempfile.TemporaryDirectory() as td:
        bank.save_dir(td)
        loaded = TaskDecoderBank.load_dir(td)
        assert 0 in loaded and 1 in loaded
        np.testing.assert_allclose(loaded.get(0).weight, bank.get(0).weight)


def test_from_batches_reconstructs() -> None:
    se, genes = _linear_data(90, se_dim=7, n_genes=4, seed=7)
    batches = [(se[i : i + 16], genes[i : i + 16]) for i in range(0, 90, 16)]
    streamed = fit_task_decoder_from_batches(
        batches, task_id=0, epochs=80, lr=5e-2
    )
    recon = streamed.decode(se)
    corr = np.corrcoef(genes.ravel(), recon.ravel())[0, 1]
    assert corr > 0.99, corr
    assert streamed.n_fit == 90


def test_mlp_fit_save_load_roundtrip() -> None:
    rng = np.random.default_rng(1)
    se = rng.normal(size=(64, 6)).astype(np.float32)
    genes = np.tanh(se @ rng.normal(size=(6, 4))).astype(np.float32)
    dec = fit_task_decoder(
        se, genes, task_id=0, epochs=40, lr=5e-2, batch_size=32, hidden=16, depth=2
    )
    assert isinstance(dec, TaskMLPDecoder)
    assert dec.se_dim == 6 and dec.n_genes == 4 and dec.hidden == 16 and dec.depth == 2
    recon = dec.decode(se)
    corr = np.corrcoef(genes.ravel(), recon.ravel())[0, 1]
    assert corr > 0.9, corr
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bank = TaskDecoderBank()
        bank.add(dec)
        bank.save_dir(td)
        loaded = TaskDecoderBank.load_dir(td)
        assert isinstance(loaded.get(0), TaskMLPDecoder)
        np.testing.assert_allclose(loaded.get(0).decode(se), recon)
        path = Path(td) / "decoder_task00.npz"
        assert isinstance(load_task_decoder(path), TaskMLPDecoder)


def test_identity_decoder_is_noop(tmp_path: Path | None = None) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5)).astype(np.float32)
    dec = identity_task_decoder(task_id=0, n_genes=5, n_fit=12)
    np.testing.assert_allclose(dec.decode(x), x.astype(np.float64))
    bank = TaskDecoderBank()
    bank.add(dec)
    np.testing.assert_allclose(bank.decode(x, np.zeros(12, dtype=np.int16)), x.astype(np.float64))
    import tempfile

    directory = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    bank.save_dir(directory)
    loaded = TaskDecoderBank.load_dir(directory)
    assert loaded.get(0).identity
    np.testing.assert_allclose(loaded.decode(x, np.zeros(12, dtype=np.int16)), x.astype(np.float64))


if __name__ == "__main__":
    test_fit_decode_roundtrip()
    test_save_load()
    test_bank_freeze_and_per_task_decode()
    test_identity_decoder_is_noop()
    test_bank_save_dir()
    test_from_batches_reconstructs()
    test_mlp_fit_save_load_roundtrip()
    print("ok")
