import warnings

import numpy as np
import polars as pl
import pytest

from phyloframe._auxlib._identity import identity
from phyloframe.legacy._impl import tgs_fit_trait, tgs_fit_traits_coarsegrain
from phyloframe.legacy._impl._tgs_minimize_nll import (
    tgs_minimize_nll,
    tgs_pick_selection_rescale,
)

from ._make_pendant_nll_kwargs import make_pendant_nll_kwargs


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


def _sample(gain_rate: float, selection_coef: float, seed: int):
    rng = np.random.default_rng(seed)
    length = np.linspace(0.5, 60.0, 120)
    num_episodes = np.full_like(length, 20_000)
    p = (
        gain_rate * length
        if selection_coef == 0
        else gain_rate * np.expm1(selection_coef * length) / selection_coef
    )
    return make_pendant_nll_kwargs(
        pl.DataFrame(
            {
                "num_episodes": num_episodes,
                "num_gains": rng.binomial(num_episodes.astype(int), p).astype(
                    float
                ),
                "mean_episode_duration": length,
            },
        )
    )


def _fit(
    nll_kwargs, *, loss_rate=0.0, fit_selection_coef_starts=(0.0, -0.3, -1.8)
):
    return tgs_fit_trait(
        fit_growth_exponent_bound=600.0,
        nll_kwargs=nll_kwargs,
        loss_rate=loss_rate,
        fit_selection_coef_starts=fit_selection_coef_starts,
    )


@pytest.mark.parametrize("gain_rate", [1e-3, 2e-3])
@pytest.mark.parametrize("selection_coef", [0.0, -0.05, -0.1])
def test_recovers_parameters(gain_rate: float, selection_coef: float):
    nll_kwargs = _sample(gain_rate, selection_coef, seed=1)
    res = _fit(nll_kwargs)
    assert res["gain_rate"] == pytest.approx(gain_rate, rel=0.1)
    assert res["selection_coef"] == pytest.approx(selection_coef, abs=0.01)
    lrt = _calc_selection_lrt(
        nll_kwargs, fit_selection_coef_starts=(0.0, -0.3, -1.8)
    )
    if selection_coef == 0:
        assert lrt < 10
    else:
        assert lrt > 25


def test_loss_rate_shifts_selection():
    nll_kwargs = _sample(1e-3, -0.05, seed=2)
    res0 = _fit(nll_kwargs)
    res1 = _fit(nll_kwargs, loss_rate=0.01)
    assert res1["selection_coef"] == pytest.approx(
        res0["selection_coef"] + 0.01, abs=1e-3
    )


def test_empty_rows_ignored():
    nll_kwargs = _sample(1e-3, -0.05, seed=3)
    expected = _fit(nll_kwargs)
    padded = make_pendant_nll_kwargs(
        pl.concat(
            [
                nll_kwargs["coarsegrain_df"],
                pl.DataFrame(
                    {
                        "num_episodes": [0.0],
                        "num_gains": [0.0],
                        "mean_episode_duration": [0.0],
                    }
                ),
            ],
        ),
    )
    assert _fit(padded) == pytest.approx(expected)


@pytest.mark.parametrize(
    "fit_selection_coef_starts", [(0.0,), (-1.8,), (-0.3, 0.0)]
)
def test_selection_starts(fit_selection_coef_starts):
    nll_kwargs = _sample(1e-3, -0.05, seed=6)
    expected = _fit(nll_kwargs)
    res = _fit(nll_kwargs, fit_selection_coef_starts=fit_selection_coef_starts)
    assert res["selection_coef"] == pytest.approx(
        expected["selection_coef"], rel=1e-3
    )
    assert res["gain_rate"] == pytest.approx(expected["gain_rate"], rel=1e-3)


def test_no_convergence_warns(monkeypatch):
    from scipy import optimize

    def fail(*args, **kwargs):
        res = optimize.minimize(*args, **kwargs)
        res.success, res.message = False, "forced failure"
        return res

    monkeypatch.setattr(
        "phyloframe.legacy._impl._tgs_minimize_nll.optimize",
        type("shim", (), {"minimize": staticmethod(fail)}),
    )
    with pytest.warns(UserWarning, match="did not converge"):
        _fit(_sample(1e-3, -0.05, seed=8))


def test_keyword_only():
    with pytest.raises(TypeError):
        tgs_fit_trait(_sample(1e-3, -0.05, seed=9), 0.0, (0.0,))


def test_no_successes_skips_fit_without_warning():
    coarsegrain_df = pl.DataFrame(
        {
            "num_episodes": [10.0],
            "num_gains": [0.0],
            "mean_episode_duration": [1.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _fit(
            dict(coarsegrain_df=coarsegrain_df),
            fit_selection_coef_starts=(0.0,),
        )
    assert res["gain_rate"] == 0.0
    assert np.isnan(res["selection_coef"])


def test_selection_bounded_by_growth_exponent_bound():
    nll_kwargs = _sample(1e-3, 0.0, seed=10)
    max_episode_duration = nll_kwargs["coarsegrain_df"][
        "mean_episode_duration"
    ].max()
    # from far past the bound, clipped into it, staying finite
    best = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        starts=[(np.log(1e-3), 1e3)],
        fit_gain=True,
        fit_selection=True,
        loss_rate=0.0,
        fit_growth_exponent_bound=5.0,
    )
    assert np.isfinite(best.fun)
    selection_coef = best.x[1] * tgs_pick_selection_rescale(
        nll_kwargs["coarsegrain_df"]
    )
    assert selection_coef * max_episode_duration <= 5.0 * (1 + 1e-9)
