import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe._auxlib._identity import identity
from phyloframe.legacy._impl import tgs_fit_traits

from ._make_tgs_fit_kwargs import make_tgs_fit_kwargs
from ._simulate_exposure_df import simulate_exposure_df


def _fit(exposure_df, weights=None, **kwargs):
    if weights is None:
        weights = np.ones(len(exposure_df))
    return tgs_fit_traits(
        weights=weights,
        progress_wrap=identity,
        trait_bin_counts_df=None,
        **make_tgs_fit_kwargs(exposure_df, **kwargs),
    )


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers(calibrate: bool):
    # four neutral traits, so each calibrates against three others
    res = _fit(simulate_exposure_df(num_neutral=4), calibrate=calibrate)
    assert res.columns == [
        "gain_rate",
        "selection_coef",
        "selection_coef_lrt_naive",
        "gain_ratio",
        "gain_ratio_lrt_naive",
    ]
    # the focal trait gains at twice the neutral rate, so its gain ratio
    # test is decisive, unlike the neutral traits'
    assert res["gain_ratio_lrt_naive"][0] > 20
    assert (res["gain_ratio_lrt_naive"][1:] < 20).all()
    assert (res["gain_ratio_lrt_naive"] > -1e-6).all()
    assert res["gain_ratio"][0] == pytest.approx(2.0, rel=0.2)
    assert res["selection_coef"][0] == pytest.approx(-0.05, abs=0.015)
    # each neutral trait's ~1,200 gains, with the selection coefficient free
    # to trade off against gain rate, leave its gain ratio uncertain by ~10%
    assert res["gain_ratio"][1:].to_list() == pytest.approx([1.0] * 4, rel=0.3)
    assert (res["selection_coef"][1:].abs() < 0.01).all()


def test_length_rtol_approximates_exact():
    exposure_df = simulate_exposure_df(8_000)
    exact = _fit(exposure_df, calibrate=True, length_rtol=0.0)
    merged = _fit(exposure_df, calibrate=True, length_rtol=1e-3)
    # absolute tolerance for likelihood-ratio statistics near 0
    pltest.assert_frame_equal(merged, exact, rel_tol=1e-3, abs_tol=1e-3)


def test_trait_without_gains():
    exposure_df = simulate_exposure_df(4_000)
    exposure_df = exposure_df.with_columns(trait0_gain=False)
    res = _fit(exposure_df)
    assert res["gain_rate"][0] == 0.0
    assert np.isnan(res["selection_coef"][0])
    assert res["gain_rate"][1:].is_finite().all()


def test_weights_match_duplication():
    exposure_df = simulate_exposure_df(4_000)
    weights = np.ones(4_000)
    weights[::3] = 2.0
    weighted = _fit(exposure_df, weights=weights)
    duplicated = _fit(pl.concat([exposure_df, exposure_df.gather_every(3)]))
    pltest.assert_frame_equal(weighted, duplicated, rel_tol=1e-4)
    assert not np.allclose(
        weighted["gain_rate"], _fit(exposure_df)["gain_rate"]
    )
