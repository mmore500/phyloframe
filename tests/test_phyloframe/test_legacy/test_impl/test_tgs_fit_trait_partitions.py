import functools

import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_calc_allmdc_data,
    tgs_calc_nll,
    tgs_fit_partitions,
    tgs_fit_trait_partitions,
    tgs_partition_allmdc_exposures,
)
from phyloframe.legacy._impl._tgs_fit import _tgs_calc_scales

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
    return data, shares, partitioning["distinct_survival_times"]


def _trait_kwargs(fixture, trait: int):
    data, shares, times = fixture
    partitions = data[data["trait"] == trait]
    calc_nll = functools.partial(
        tgs_calc_nll,
        partitions=partitions,
        growth_exponent_bound=600.0,
        survival_shares=shares[partitions.index],
        distinct_survival_times=times,
    )
    return dict(calc_nll=calc_nll, partitions=partitions)


def test_calc_scales(fixture):
    partitions = _trait_kwargs(fixture, 0)["partitions"]
    n, k, x = (
        partitions[c] for c in ("num_trials", "num_successes", "length")
    )
    selection_scale, log_gain_start = _tgs_calc_scales(partitions)
    assert selection_scale == pytest.approx(n.sum() / (n * x).sum())
    assert log_gain_start == pytest.approx(np.log(k.sum() / (n * x).sum()))


def test_true_parameters_fit_better(fixture):
    calc_nll = _trait_kwargs(fixture, 0)["calc_nll"]
    nll, dlog_gain, drate = calc_nll(log_gain_rate=np.log(2e-3), rate=-0.05)
    assert np.isfinite([nll, dlog_gain, drate]).all()
    # the true parameters fit better than a far-off gain rate
    assert nll < calc_nll(log_gain_rate=np.log(2e-2), rate=-0.05)[0]


def test_recovers_focal(fixture):
    res = tgs_fit_partitions(
        **_trait_kwargs(fixture, 0),
        loss_rate=0.0,
        selection_starts=(0.0, -0.3, -1.8),
    )
    assert res.gain_rate == pytest.approx(2e-3, rel=0.15)
    assert res.selection == pytest.approx(-0.05, abs=0.01)
    assert res.selection_lrt_naive > 10


def test_recovers_neutral(fixture):
    res = tgs_fit_partitions(
        **_trait_kwargs(fixture, 1), loss_rate=0.0, selection_starts=(0.0,)
    )
    assert res.gain_rate == pytest.approx(1e-3, rel=0.15)
    assert abs(res.selection) < 0.01


def test_loss_rate_shifts_selection(fixture):
    kwargs = dict(**_trait_kwargs(fixture, 0), selection_starts=(0.0, -0.3))
    res = tgs_fit_partitions(loss_rate=0.0, **kwargs)
    lossy = tgs_fit_partitions(loss_rate=0.01, **kwargs)
    # fitted rate, selection - loss rate, is unchanged
    assert lossy.selection - 0.01 == pytest.approx(res.selection, abs=1e-4)


def test_fit_trait_partitions(fixture):
    data, shares, times = fixture
    res = tgs_fit_trait_partitions(
        data=data,
        neutral_traits=[1, 2],
        loss_rate=0.0,
        selection_starts=(0.0, -0.3),
        growth_exponent_bound=600.0,
        survival_shares=shares,
        distinct_survival_times=times,
    )
    assert list(res.columns) == [
        "gain_rate",
        "selection",
        "selection_lrt_naive",
        "gain_ratio",
        "gain_ratio_lrt_naive",
    ]
    assert len(res) == 3
    # matches fitting each trait alone
    for trait in range(3):
        alone = tgs_fit_partitions(
            **_trait_kwargs(fixture, trait),
            loss_rate=0.0,
            selection_starts=(0.0, -0.3),
        )
        assert res.loc[trait, "gain_rate"] == pytest.approx(alone.gain_rate)
        assert res.loc[trait, "selection"] == pytest.approx(alone.selection)
    # focal gains at twice the neutral rate
    assert res.loc[0, "gain_ratio"] == pytest.approx(2.0, rel=0.2)
    assert res.loc[0, "gain_ratio_lrt_naive"] > 10
