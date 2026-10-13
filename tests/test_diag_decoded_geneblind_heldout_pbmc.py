import numpy as np

from experiments.tx.diag_decoded_geneblind_heldout_pbmc import (
    GroupMean,
    average_source_effects,
    pool_group_means_by_cell_type,
    residualize_within_pools,
)


def test_residualize_within_target_donor_cell_type_pools() -> None:
    values = np.asarray([[1.0], [3.0], [10.0], [14.0], [18.0]])
    pools = [("D1", "T"), ("D1", "T"), ("D1", "B"), ("D1", "B"), ("D1", "B")]

    got = residualize_within_pools(values, pools)

    np.testing.assert_allclose(got[:, 0], [-1.0, 1.0, -4.0, 0.0, 4.0])


def test_average_source_effects_balances_donors_then_cytokines() -> None:
    effects = {
        ("T0a", "T", "C1"): np.asarray([1.0, 2.0]),
        ("T0b", "T", "C1"): np.asarray([3.0, 4.0]),
        ("T0a", "T", "C2"): np.asarray([10.0, 20.0]),
    }

    b4, b3 = average_source_effects(effects)

    np.testing.assert_allclose(b4[("T", "C1")], [2.0, 3.0])
    np.testing.assert_allclose(b4[("T", "C2")], [10.0, 20.0])
    np.testing.assert_allclose(b3["T"], [6.0, 11.5])


def test_pool_group_means_by_cell_type_is_cell_weighted() -> None:
    means = {
        ("D1", "T", "C1"): GroupMean(
            z=np.asarray([1.0]), x=np.asarray([10.0]), n=1
        ),
        ("D2", "T", "C1"): GroupMean(
            z=np.asarray([4.0]), x=np.asarray([40.0]), n=3
        ),
    }

    got = pool_group_means_by_cell_type(means)[("pooled", "T", "C1")]

    assert got.n == 4
    np.testing.assert_allclose(got.z, [3.25])
    np.testing.assert_allclose(got.x, [32.5])
