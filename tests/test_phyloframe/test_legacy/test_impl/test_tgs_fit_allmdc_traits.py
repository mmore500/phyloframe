import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_fit_allmdc_traits,
    tgs_partition_allmdc_branches,
)

from ._simulate_allmdc_branches import simulate_allmdc_branches


def _fit(branches, calibrate=False, unit_weights=None):
    num_traits = sum(c.endswith("_at_risk") for c in branches)
    neutral = list(range(1, num_traits))
    partitioning = tgs_partition_allmdc_branches(
        branches=branches,
        neutral_traits=neutral,
        trait_names=list(map(str, range(num_traits))),
        calibrate=calibrate,
        calibrate_neighborhood_fractions=(0.5, 1.0),
        calibrate_num_partitions=256,
        calibrate_size_partition_ratio=4.0,
        calibrate_size_partition_min_neutral_gains=1,
        num_survival_time_bins=32,
    )
    return tgs_fit_allmdc_traits(
        branches=branches,
        partitioning=partitioning,
        unit_weights=(
            np.ones(len(branches)) if unit_weights is None else unit_weights
        ),
        neutral_traits=neutral,
        calibrate=calibrate,
        loss_rate=0.0,
        selection_starts=(0.0, -0.3, -1.8),
        max_growth_exponent=600.0,
    )


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers(calibrate: bool):
    # four neutral traits, so each calibrates against three others
    res = _fit(simulate_allmdc_branches(num_neutral=4), calibrate=calibrate)
    assert list(res.columns) == [
        "gain_rate",
        "selection",
        "selection_lrt_naive",
        "gain_ratio",
    ]
    assert res.loc[0, "gain_ratio"] == pytest.approx(2.0, rel=0.2)
    assert res.loc[0, "selection"] == pytest.approx(-0.05, abs=0.015)
    # each neutral trait's ~1,200 gains, with selection free to trade off
    # against gain rate, leave its gain ratio uncertain by ~10%
    assert res.loc[1:, "gain_ratio"].tolist() == pytest.approx(
        [1.0] * 4, rel=0.3
    )
    assert (res.loc[1:, "selection"].abs() < 0.01).all()


def test_trait_without_gains():
    branches = simulate_allmdc_branches(4_000)
    branches = branches.assign(trait0_gain=False)
    res = _fit(branches)
    assert res.loc[0, "gain_rate"] == 0.0
    assert np.isnan(res.loc[0, "selection"])
    assert np.isfinite(res.loc[1:, "gain_rate"]).all()


def test_weights_match_duplication():
    branches = simulate_allmdc_branches(4_000)
    weights = np.ones(4_000)
    weights[::3] = 2.0
    weighted = _fit(branches, unit_weights=weights)
    assert not np.allclose(weighted["gain_rate"], _fit(branches)["gain_rate"])
