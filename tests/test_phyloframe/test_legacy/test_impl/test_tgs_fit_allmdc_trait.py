import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_calc_allmdc_data,
    tgs_fit_allmdc_trait,
    tgs_make_allmdc_trait_nll,
    tgs_partition_allmdc_exposures,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


@pytest.fixture(scope="module")
def fixture():
    exposure_df = simulate_allmdc_exposure_df()
    partitioning = tgs_partition_allmdc_exposures(
        exposure_df=exposure_df,
        neutral_traits=[1, 2],
        trait_names=["focal", "n1", "n2"],
        calibrate=False,
        calibrate_neighborhood_fractions=(1.0,),
        calibrate_num_partitions=256,
        calibrate_size_partition_ratio=4.0,
        calibrate_size_partition_min_neutral_gains=1,
    )
    data, shares = tgs_calc_allmdc_data(
        exposure_df=exposure_df,
        partitioning=partitioning,
        weights=np.ones(len(exposure_df)),
        neutral_traits=[1, 2],
        calibrate=False,
    )
    by_trait = [
        dict(partitions=partitions, survival_shares=shares[partitions.index])
        for __, partitions in data.groupby("trait", observed=False)
    ]
    return by_trait, partitioning["distinct_survival_times"]


def _fit(
    *,
    partitions,
    survival_shares,
    distinct_survival_times,
    growth_exponent_bound,
    **kwargs,
):
    calc_nll, selection_scale = tgs_make_allmdc_trait_nll(
        partitions=partitions,
        survival_shares=survival_shares,
        distinct_survival_times=distinct_survival_times,
        growth_exponent_bound=growth_exponent_bound,
    )
    return tgs_fit_allmdc_trait(
        calc_nll=calc_nll,
        selection_scale=selection_scale,
        partitions=partitions,
        **kwargs,
    )


def test_make_nll(fixture):
    data, times = fixture
    calc_nll, selection_scale = tgs_make_allmdc_trait_nll(
        **data[0], distinct_survival_times=times, growth_exponent_bound=600.0
    )
    partitions = data[0]["partitions"]
    n, x = partitions["num_trials"], partitions["length"]
    assert selection_scale == pytest.approx(n.sum() / (n * x).sum())
    nll, dlog_gain, drate = calc_nll(np.log(2e-3), -0.05)
    assert np.isfinite([nll, dlog_gain, drate]).all()
    # the true parameters fit better than a far-off gain rate
    assert nll < calc_nll(np.log(2e-2), -0.05)[0]


def test_recovers_focal(fixture):
    data, times = fixture
    res = _fit(
        **data[0],
        distinct_survival_times=times,
        loss_rate=0.0,
        selection_starts=(0.0, -0.3, -1.8),
        growth_exponent_bound=600.0,
    )
    assert res["gain_rate"] == pytest.approx(2e-3, rel=0.15)
    assert res["selection"] == pytest.approx(-0.05, abs=0.01)
    assert res["selection_lrt_naive"] > 10


def test_recovers_neutral(fixture):
    data, times = fixture
    res = _fit(
        **data[1],
        distinct_survival_times=times,
        loss_rate=0.0,
        selection_starts=(0.0,),
        growth_exponent_bound=600.0,
    )
    assert res["gain_rate"] == pytest.approx(1e-3, rel=0.15)
    assert abs(res["selection"]) < 0.01


def test_loss_rate_shifts_selection(fixture):
    data, times = fixture
    kwargs = dict(
        **data[0],
        distinct_survival_times=times,
        selection_starts=(0.0, -0.3),
        growth_exponent_bound=600.0,
    )
    res = _fit(loss_rate=0.0, **kwargs)
    lossy = _fit(loss_rate=0.01, **kwargs)
    # fitted rate, selection - loss rate, is unchanged
    assert lossy["selection"] - 0.01 == pytest.approx(
        res["selection"], abs=1e-4
    )
