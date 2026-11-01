import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_calc_allmdc_data,
    tgs_fit_allmdc_trait,
    tgs_partition_allmdc_branches,
)

from ._simulate_allmdc_branches import simulate_allmdc_branches


@pytest.fixture(scope="module")
def fixture():
    branches = simulate_allmdc_branches()
    partitioning = tgs_partition_allmdc_branches(
        branches=branches,
        neutral_traits=[1, 2],
        trait_names=["focal", "n1", "n2"],
        calibrate=False,
        calibrate_neighborhood_fractions=(1.0,),
        calibrate_num_partitions=256,
        calibrate_size_partition_ratio=4.0,
        calibrate_size_partition_min_neutral_gains=1,
        num_survival_time_bins=32,
    )
    data = tgs_calc_allmdc_data(
        branches=branches,
        partitioning=partitioning,
        unit_weights=np.ones(len(branches)),
        neutral_traits=[1, 2],
        calibrate=False,
    )
    return data, partitioning["survival_bin_times"]


def test_recovers_focal(fixture):
    data, times = fixture
    res = tgs_fit_allmdc_trait(
        data=data[0],
        survival_bin_times=times,
        loss_rate=0.0,
        selection_starts=(0.0, -0.3, -1.8),
        growth_exponent_bound=600.0,
    )
    assert res["gain_rate"] == pytest.approx(2e-3, rel=0.15)
    assert res["selection"] == pytest.approx(-0.05, abs=0.01)
    assert res["selection_lrt_naive"] > 10


def test_recovers_neutral(fixture):
    data, times = fixture
    res = tgs_fit_allmdc_trait(
        data=data[1],
        survival_bin_times=times,
        loss_rate=0.0,
        selection_starts=(0.0,),
        growth_exponent_bound=600.0,
    )
    assert res["gain_rate"] == pytest.approx(1e-3, rel=0.15)
    assert abs(res["selection"]) < 0.01


def test_loss_rate_shifts_selection(fixture):
    data, times = fixture
    kwargs = dict(
        data=data[0],
        survival_bin_times=times,
        selection_starts=(0.0, -0.3),
        growth_exponent_bound=600.0,
    )
    res = tgs_fit_allmdc_trait(loss_rate=0.0, **kwargs)
    lossy = tgs_fit_allmdc_trait(loss_rate=0.01, **kwargs)
    # fitted rate, selection - loss rate, is unchanged
    assert lossy["selection"] - 0.01 == pytest.approx(
        res["selection"], abs=1e-4
    )
