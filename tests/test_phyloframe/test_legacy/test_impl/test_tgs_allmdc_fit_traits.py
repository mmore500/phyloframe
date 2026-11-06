import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_allmdc_fit_traits,
    tgs_allmdc_partition_episodes,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def _fit(exposure_df, calibrate=False, weights=None):
    num_traits = sum(c.endswith("_starts_absent") for c in exposure_df)
    neutral = list(range(1, num_traits))
    partitioning = tgs_allmdc_partition_episodes(
        exposure_df=exposure_df,
        neutral_traits=neutral,
        trait_names=list(map(str, range(num_traits))),
        calibrate=calibrate,
        calibrate_neighborhood_fractions=(0.5, 1.0),
        calibrate_num_partitions=256,
        calibrate_max_rate_ratio=np.inf,
        calibrate_size_partition_ratio=4.0,
        calibrate_size_partition_min_neutral_gains=1,
    )
    return tgs_allmdc_fit_traits(
        exposure_df=exposure_df,
        partitioning=partitioning,
        weights=(np.ones(len(exposure_df)) if weights is None else weights),
        neutral_traits=neutral,
        calibrate=calibrate,
        calibrate_max_rate_ratio=np.inf,
        loss_rate=0.0,
        selection_coef_starts=(0.0, -0.3, -1.8),
        growth_exponent_bound=600.0,
    )


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers(calibrate: bool):
    # four neutral traits, so each calibrates against three others
    res = _fit(simulate_allmdc_exposure_df(num_neutral=4), calibrate=calibrate)
    assert list(res.columns) == [
        "gain_rate",
        "selection_coef",
        "selection_coef_lrt_naive",
        "gain_ratio",
        "gain_ratio_lrt_naive",
    ]
    # the focal trait gains at twice the neutral rate, so its gain ratio
    # test is decisive, unlike the neutral traits'
    assert res.loc[0, "gain_ratio_lrt_naive"] > 20
    assert (res.loc[1:, "gain_ratio_lrt_naive"] < 20).all()
    assert (res["gain_ratio_lrt_naive"] > -1e-6).all()
    assert res.loc[0, "gain_ratio"] == pytest.approx(2.0, rel=0.2)
    assert res.loc[0, "selection_coef"] == pytest.approx(-0.05, abs=0.015)
    # each neutral trait's ~1,200 gains, with the selection coefficient free to
    # trade off against gain rate, leave its gain ratio uncertain by ~10%
    assert res.loc[1:, "gain_ratio"].tolist() == pytest.approx(
        [1.0] * 4, rel=0.3
    )
    assert (res.loc[1:, "selection_coef"].abs() < 0.01).all()


def test_trait_without_gains():
    exposure_df = simulate_allmdc_exposure_df(4_000)
    exposure_df = exposure_df.assign(trait0_gain=False)
    res = _fit(exposure_df)
    assert res.loc[0, "gain_rate"] == 0.0
    assert np.isnan(res.loc[0, "selection_coef"])
    assert np.isfinite(res.loc[1:, "gain_rate"]).all()


def test_weights_match_duplication():
    exposure_df = simulate_allmdc_exposure_df(4_000)
    weights = np.ones(4_000)
    weights[::3] = 2.0
    weighted = _fit(exposure_df, weights=weights)
    assert not np.allclose(
        weighted["gain_rate"], _fit(exposure_df)["gain_rate"]
    )
