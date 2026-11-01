import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_calc_allmdc_data,
    tgs_partition_allmdc_branches,
)

from ._simulate_allmdc_branches import simulate_allmdc_branches

_KEYS = ("num_trials", "num_successes", "lengths", "survival_shares")


def _partition(branches, calibrate):
    return tgs_partition_allmdc_branches(
        branches=branches,
        neutral_traits=[1, 2],
        trait_names=["focal", "n1", "n2"],
        calibrate=calibrate,
        calibrate_neighborhood_fractions=(0.25, 1.0),
        calibrate_num_partitions=64,
        calibrate_size_partition_ratio=4.0,
        calibrate_size_partition_min_neutral_gains=1,
        num_survival_time_bins=8,
    )


@pytest.mark.parametrize("calibrate", [False, True])
def test_data(calibrate: bool):
    branches = simulate_allmdc_branches(4_000)
    partitioning = _partition(branches, calibrate)
    data = tgs_calc_allmdc_data(
        branches=branches,
        partitioning=partitioning,
        unit_weights=np.ones(4_000),
        neutral_traits=[1, 2],
        calibrate=calibrate,
    )
    assert len(data) == 3
    for trait, trait_data in enumerate(data):
        n, k, lengths, shares = map(trait_data.get, _KEYS)
        assert n.sum() == 4_000
        assert k.sum() == branches[f"trait{trait}_gain"].sum()
        assert (lengths > 0).all()
        assert shares.shape == (
            len(n),
            len(partitioning["survival_bin_times"]),
        )
        np.testing.assert_allclose(np.asarray(shares.sum(axis=1)).ravel(), 1.0)
    if not calibrate:
        n, __, lengths, __ = map(data[0].get, _KEYS)
        assert (n * lengths).sum() == pytest.approx(
            branches["branch_length"].sum()
        )


def test_weights():
    branches = simulate_allmdc_branches(4_000)
    partitioning = _partition(branches, False)
    weights = np.zeros(4_000)
    weights[:100] = 2.0
    trait_data, *__ = tgs_calc_allmdc_data(
        branches=branches,
        partitioning=partitioning,
        unit_weights=weights,
        neutral_traits=[1, 2],
        calibrate=False,
    )
    n, shares = trait_data["num_trials"], trait_data["survival_shares"]
    # partitions without weighted branches are omitted
    assert n.sum() == 200
    assert (n > 0).all()
    np.testing.assert_allclose(np.asarray(shares.sum(axis=1)).ravel(), 1.0)


def test_trait_without_units():
    branches = simulate_allmdc_branches(1_000)
    branches = branches.assign(trait0_starts_absent=False, trait0_gain=False)
    trait_data, *__ = tgs_calc_allmdc_data(
        branches=branches,
        partitioning=_partition(branches, False),
        unit_weights=np.ones(1_000),
        neutral_traits=[1, 2],
        calibrate=False,
    )
    n, k, lengths, shares = map(trait_data.get, _KEYS)
    assert len(n) == len(k) == len(lengths) == shares.shape[0] == 0
