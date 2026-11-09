import warnings

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._impl._tgs_fit_coarsened_traits import (
    _tgs_calc_trait_selection_lrt,
)
from phyloframe.legacy._impl._tgs_fit_trait import tgs_fit_trait
from phyloframe.legacy._impl._tgs_minimize_nll import tgs_minimize_nll

from ._make_pendant_nll_kwargs import make_pendant_nll_kwargs


def _sample(gain_rate: float, selection_coef: float, seed: int):
    rng = np.random.default_rng(seed)
    length = np.linspace(0.5, 60.0, 120)
    num_trials = np.full_like(length, 20_000)
    p = (
        gain_rate * length
        if selection_coef == 0
        else gain_rate * np.expm1(selection_coef * length) / selection_coef
    )
    return make_pendant_nll_kwargs(
        pl.DataFrame(
            {
                "num_trials": num_trials,
                "num_successes": rng.binomial(
                    num_trials.astype(int), p
                ).astype(float),
                "length": length,
            },
        )
    )


def _fit(
    nll_kwargs, *, loss_rate=0.0, selection_coef_starts=(0.0, -0.3, -1.8)
):
    return tgs_fit_trait(
        growth_exponent_bound=600.0,
        nll_kwargs=nll_kwargs,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
    )


@pytest.mark.parametrize("gain_rate", [1e-3, 2e-3])
@pytest.mark.parametrize("selection_coef", [0.0, -0.05, -0.1])
def test_recovers_parameters(gain_rate: float, selection_coef: float):
    nll_kwargs = _sample(gain_rate, selection_coef, seed=1)
    res = _fit(nll_kwargs)
    assert res["gain_rate"] == pytest.approx(gain_rate, rel=0.1)
    assert res["selection_coef"] == pytest.approx(selection_coef, abs=0.01)
    lrt = _tgs_calc_trait_selection_lrt(
        growth_exponent_bound=600.0,
        nll_kwargs=nll_kwargs,
        loss_rate=0.0,
        **res,
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
                nll_kwargs["coarsened_df"],
                pl.DataFrame(
                    {
                        "num_trials": [0.0],
                        "num_successes": [0.0],
                        "length": [0.0],
                    }
                ),
            ],
        ),
    )
    assert _fit(padded) == pytest.approx(expected)


@pytest.mark.parametrize(
    "selection_coef_starts", [(0.0,), (-1.8,), (-0.3, 0.0)]
)
def test_selection_starts(selection_coef_starts):
    nll_kwargs = _sample(1e-3, -0.05, seed=6)
    expected = _fit(nll_kwargs)
    res = _fit(nll_kwargs, selection_coef_starts=selection_coef_starts)
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
    coarsened_df = pl.DataFrame(
        {"num_trials": [10.0], "num_successes": [0.0], "length": [1.0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _fit(
            dict(coarsened_df=coarsened_df), selection_coef_starts=(0.0,)
        )
    assert res["gain_rate"] == 0.0
    assert np.isnan(res["selection_coef"])


def test_selection_bounded_by_growth_exponent_bound():
    nll_kwargs = _sample(1e-3, 0.0, seed=10)
    max_length = nll_kwargs["coarsened_df"]["length"].max()
    # from far past the bound, clipped into it, staying finite
    best = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        starts=[(np.log(1e-3), 1e3)],
        fit_gain=True,
        fit_selection=True,
        selection_rescale=1.0,
        loss_rate=0.0,
        growth_exponent_bound=5.0,
    )
    assert np.isfinite(best.fun)
    assert best.x[1] * max_length <= 5.0 * (1 + 1e-9)
