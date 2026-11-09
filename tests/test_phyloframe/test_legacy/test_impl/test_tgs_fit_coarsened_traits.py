import numpy as np
import pytest

from phyloframe._auxlib._identity import identity
from phyloframe.legacy._impl import (
    tgs_calc_nll,
    tgs_fit_coarsened_traits,
    tgs_fit_trait,
)
from phyloframe.legacy._impl._tgs_fit_coarsened_traits import (
    _tgs_calc_lrt,
    _tgs_calc_trait_selection_lrt,
)
from phyloframe.legacy._impl._tgs_fit_traits import _tgs_coarsen_episodes
from phyloframe.legacy._impl._tgs_minimize_nll import (
    tgs_pick_log_gain_rate_start,
    tgs_pick_selection_rescale,
)

from ._simulate_exposure_df import simulate_exposure_df


@pytest.fixture(scope="module")
def fixture():
    exposure_df = simulate_exposure_df()
    distinct_survival_durations, survival_duration_indices = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    # each trait's `tgs_calc_nll` arguments, but the rate parameters
    trait_nll_kwargs = []
    for trait in range(3):
        coarsened_df, shares = _tgs_coarsen_episodes(
            exposure_df=exposure_df,
            trait_id=trait,
            bins=np.zeros(len(exposure_df), dtype=np.int64),
            calibration_factors=np.ones(1),
            weights=np.ones(len(exposure_df)),
            length_rtol=1e-2,
            survival_duration_indices=survival_duration_indices,
            num_survival_durations=len(distinct_survival_durations),
        )
        trait_nll_kwargs.append(
            dict(
                coarsened_df=coarsened_df,
                growth_exponent_bound=600.0,
                survival_shares=shares,
                distinct_survival_durations=distinct_survival_durations,
            ),
        )
    return trait_nll_kwargs


def _nll_kwargs(fixture, trait: int):
    return fixture[trait]


def test_calc_scales(fixture):
    coarsened_df = _nll_kwargs(fixture, 0)["coarsened_df"]
    n, k, x = (
        coarsened_df[c] for c in ("num_trials", "num_successes", "length")
    )
    selection_rescale = tgs_pick_selection_rescale(coarsened_df)
    log_gain_start = tgs_pick_log_gain_rate_start(coarsened_df)
    assert selection_rescale == pytest.approx(n.sum() / (n * x).sum())
    assert log_gain_start == pytest.approx(np.log(k.sum() / (n * x).sum()))


def test_true_parameters_fit_better(fixture):
    nll_kwargs = _nll_kwargs(fixture, 0)
    nll, dlog_gain, drate = tgs_calc_nll(
        **nll_kwargs, gain_rate=2e-3, growth_rate=-0.05
    )
    assert np.isfinite([nll, dlog_gain, drate]).all()
    # the true parameters fit better than a far-off gain rate
    assert (
        nll < tgs_calc_nll(**nll_kwargs, gain_rate=2e-2, growth_rate=-0.05)[0]
    )


def test_recovers_focal(fixture):
    nll_kwargs = _nll_kwargs(fixture, 0)
    res = tgs_fit_trait(
        nll_kwargs=nll_kwargs,
        loss_rate=0.0,
        selection_coef_starts=(0.0, -0.3, -1.8),
    )
    assert res["gain_rate"] == pytest.approx(2e-3, rel=0.15)
    assert res["selection_coef"] == pytest.approx(-0.05, abs=0.01)
    lrt = _tgs_calc_trait_selection_lrt(
        nll_kwargs=nll_kwargs, loss_rate=0.0, **res
    )
    assert lrt > 10


def test_recovers_neutral(fixture):
    nll_kwargs = _nll_kwargs(fixture, 1)
    res = tgs_fit_trait(
        nll_kwargs=nll_kwargs,
        loss_rate=0.0,
        selection_coef_starts=(0.0,),
    )
    assert res["gain_rate"] == pytest.approx(1e-3, rel=0.15)
    assert abs(res["selection_coef"]) < 0.01
    lrt = _tgs_calc_trait_selection_lrt(
        nll_kwargs=nll_kwargs, loss_rate=0.0, **res
    )
    assert -1e-6 < lrt < 10


def test_loss_rate_shifts_selection(fixture):
    kwargs = dict(
        nll_kwargs=_nll_kwargs(fixture, 0), selection_coef_starts=(0.0, -0.3)
    )
    res = tgs_fit_trait(loss_rate=0.0, **kwargs)
    lossy = tgs_fit_trait(loss_rate=0.01, **kwargs)
    # fitted rate, selection coefficient - loss rate, is unchanged
    assert lossy["selection_coef"] - 0.01 == pytest.approx(
        res["selection_coef"], abs=1e-4
    )


def test_fit_coarsened_traits(fixture):
    res = tgs_fit_coarsened_traits(
        trait_nll_kwargs=fixture,
        neutral_traits=[1, 2],
        loss_rate=0.0,
        selection_coef_starts=(0.0, -0.3),
        progress_wrap=identity,
    )
    assert res.columns == [
        "gain_rate",
        "selection_coef",
        "selection_coef_lrt_naive",
        "gain_ratio",
        "gain_ratio_lrt_naive",
    ]
    assert len(res) == 3
    # matches fitting each trait alone
    for trait in range(3):
        nll_kwargs = _nll_kwargs(fixture, trait)
        alone = tgs_fit_trait(
            nll_kwargs=nll_kwargs,
            loss_rate=0.0,
            selection_coef_starts=(0.0, -0.3),
        )
        assert res["gain_rate"][trait] == pytest.approx(alone["gain_rate"])
        assert res["selection_coef"][trait] == pytest.approx(
            alone["selection_coef"]
        )
        assert res["selection_coef_lrt_naive"][trait] == pytest.approx(
            _tgs_calc_trait_selection_lrt(
                nll_kwargs=nll_kwargs, loss_rate=0.0, **alone
            )
        )
    # focal gains at twice the neutral rate
    assert res["gain_ratio"][0] == pytest.approx(2.0, rel=0.2)
    assert res["gain_ratio_lrt_naive"][0] > 10


def test_negative_lrt_clipped_with_warning():
    assert _tgs_calc_lrt(null_nll=101.5, free_nll=100.0) == 3.0
    # within optimizer tolerance, quietly
    assert _tgs_calc_lrt(null_nll=100.0 - 1e-9, free_nll=100.0) == 0.0
    with pytest.warns(UserWarning, match="clipping"):
        assert _tgs_calc_lrt(null_nll=99.5, free_nll=100.0) == 0.0
