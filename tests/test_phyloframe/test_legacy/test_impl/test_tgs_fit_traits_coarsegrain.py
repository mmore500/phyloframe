import numpy as np
import pytest
from scipy import optimize

from phyloframe._auxlib._identity import identity
from phyloframe.legacy._impl import (
    tgs_calc_nll,
    tgs_fit_trait,
    tgs_fit_traits_coarsegrain,
)
from phyloframe.legacy._impl._tgs_fit_traits import _tgs_coarsegrain_episodes
from phyloframe.legacy._impl._tgs_fit_traits_coarsegrain import _tgs_calc_lrt
from phyloframe.legacy._impl._tgs_minimize_nll import (
    tgs_pick_selection_rescale,
)

from ._simulate_exposure_df import simulate_exposure_df


def _calc_selection_lrt(nll_kwargs, **kwargs) -> float:
    """One trait's selection likelihood-ratio statistic, with copies of
    itself as its neutral reference."""
    return tgs_fit_traits_coarsegrain(
        trait_nll_kwargs=[nll_kwargs] * 3,
        neutral_trait_ids=[1, 2],
        loss_rate=0.0,
        fit_growth_exponent_bound=600.0,
        calc_lrts=True,
        progress_wrap=identity,
        **kwargs,
    )["selection_coef_lrt_naive"][0]


@pytest.fixture(scope="module")
def fixture():
    exposure_df = simulate_exposure_df()
    distinct_survival_durations, survival_duration_ids = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    exposure_df = exposure_df.with_columns(
        weight=1.0, survival_duration_id=survival_duration_ids
    )
    # each trait's `tgs_calc_nll` arguments, but the rate parameters
    trait_nll_kwargs = []
    for trait in range(3):
        coarsegrain_df, shares = _tgs_coarsegrain_episodes(
            exposure_df,
            trait_id=trait,
            calibration_factors=np.ones(1),
            nll_coarsegrain_rtol=1e-2,
            num_survival_durations=len(distinct_survival_durations),
        )
        trait_nll_kwargs.append(
            dict(
                coarsegrain_df=coarsegrain_df,
                survival_duration_granule_mass=shares,
                distinct_survival_durations=distinct_survival_durations,
            ),
        )
    return trait_nll_kwargs


def _nll_kwargs(fixture, trait: int):
    return fixture[trait]


def test_calc_scales(fixture):
    coarsegrain_df = _nll_kwargs(fixture, 0)["coarsegrain_df"]
    n, x = (
        coarsegrain_df["num_episodes"],
        coarsegrain_df["mean_episode_duration"],
    )
    selection_rescale = tgs_pick_selection_rescale(coarsegrain_df)
    assert selection_rescale == pytest.approx(n.sum() / (n * x).sum())


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
        fit_growth_exponent_bound=600.0,
        nll_kwargs=nll_kwargs,
        loss_rate=0.0,
        fit_selection_coef_starts=(0.0, -0.3, -1.8),
    )
    assert res["gain_rate"] == pytest.approx(2e-3, rel=0.15)
    assert res["selection_coef"] == pytest.approx(-0.05, abs=0.01)
    lrt = _calc_selection_lrt(
        nll_kwargs, fit_selection_coef_starts=(0.0, -0.3, -1.8)
    )
    assert lrt > 10


def test_recovers_neutral(fixture):
    nll_kwargs = _nll_kwargs(fixture, 1)
    res = tgs_fit_trait(
        fit_growth_exponent_bound=600.0,
        nll_kwargs=nll_kwargs,
        loss_rate=0.0,
        fit_selection_coef_starts=(0.0,),
    )
    assert res["gain_rate"] == pytest.approx(1e-3, rel=0.15)
    assert abs(res["selection_coef"]) < 0.01
    lrt = _calc_selection_lrt(nll_kwargs, fit_selection_coef_starts=(0.0,))
    assert -1e-6 < lrt < 10


def test_loss_rate_shifts_selection(fixture):
    kwargs = dict(
        nll_kwargs=_nll_kwargs(fixture, 0),
        fit_selection_coef_starts=(0.0, -0.3),
    )
    res = tgs_fit_trait(
        loss_rate=0.0, fit_growth_exponent_bound=600.0, **kwargs
    )
    lossy = tgs_fit_trait(
        loss_rate=0.01, fit_growth_exponent_bound=600.0, **kwargs
    )
    # fitted rate, selection coefficient - loss rate, is unchanged
    assert lossy["selection_coef"] - 0.01 == pytest.approx(
        res["selection_coef"], abs=1e-4
    )


def test_fit_traits_coarsegrain(fixture):
    res = tgs_fit_traits_coarsegrain(
        fit_growth_exponent_bound=600.0,
        trait_nll_kwargs=fixture,
        neutral_trait_ids=[1, 2],
        loss_rate=0.0,
        fit_selection_coef_starts=(0.0, -0.3),
        calc_lrts=True,
        progress_wrap=identity,
    )
    assert res.columns == [
        "gain_rate",
        "selection_coef",
        "gain_ratio",
        "selection_coef_lrt_naive",
        "gain_ratio_lrt_naive",
    ]
    assert len(res) == 3
    # matches fitting each trait alone
    for trait in range(3):
        nll_kwargs = _nll_kwargs(fixture, trait)
        alone = tgs_fit_trait(
            fit_growth_exponent_bound=600.0,
            nll_kwargs=nll_kwargs,
            loss_rate=0.0,
            fit_selection_coef_starts=(0.0, -0.3),
        )
        assert res["gain_rate"][trait] == pytest.approx(alone["gain_rate"])
        assert res["selection_coef"][trait] == pytest.approx(
            alone["selection_coef"]
        )
        assert res["selection_coef_lrt_naive"][trait] == pytest.approx(
            _calc_selection_lrt(
                nll_kwargs, fit_selection_coef_starts=(0.0, -0.3)
            )
        )
    # focal gains at twice the neutral rate
    assert res["gain_ratio"][0] == pytest.approx(2.0, rel=0.2)
    assert res["gain_ratio_lrt_naive"][0] > 10


def test_negative_lrt_clipped_with_warning(monkeypatch):
    def calc_lrt(null_nll: float) -> float:
        monkeypatch.setattr(
            "phyloframe.legacy._impl._tgs_fit_traits_coarsegrain."
            "tgs_minimize_nll",
            lambda: optimize.OptimizeResult(fun=null_nll, num_converged=1),
        )
        return _tgs_calc_lrt(free_nll=100.0)

    assert calc_lrt(101.5) == 3.0
    # within optimizer tolerance, quietly
    assert calc_lrt(100.0 - 1e-9) == 0.0
    with pytest.warns(UserWarning, match="unreliable"):
        assert calc_lrt(99.5) == 0.0
