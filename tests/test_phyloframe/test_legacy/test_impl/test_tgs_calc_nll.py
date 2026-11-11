from fractions import Fraction
from math import factorial

import numpy as np
import polars as pl
import pytest
from scipy import optimize

from phyloframe.legacy._impl import tgs_calc_nll
from phyloframe.legacy._impl._tgs_calc_nll import _tgs_calc_effective_length

from ._make_pendant_nll_kwargs import make_pendant_nll_kwargs


def _data():
    rng = np.random.default_rng(1)
    n = rng.integers(50, 100, 6).astype(float)
    k = rng.integers(0, 5, 6).astype(float)
    lengths = rng.uniform(1, 20, 6)
    shares = rng.random((6, 3))
    shares /= shares.sum(axis=1, keepdims=True)
    coarsegrain_df = pl.DataFrame(
        {"num_episodes": n, "num_gains": k, "mean_episode_duration": lengths}
    )
    data = dict(
        coarsegrain_df=coarsegrain_df, survival_duration_granule_mass=shares
    )
    return data, np.array([0.0, 5.0, 30.0])


def _pendant_coarsegrain_df(
    gain_rate: float, selection_coef: float, seed: int
):
    rng = np.random.default_rng(seed)
    length = np.linspace(0.5, 60.0, 120)
    num_episodes = np.full_like(length, 20_000)
    p = (
        gain_rate * length
        if selection_coef == 0
        else gain_rate * np.expm1(selection_coef * length) / selection_coef
    )
    return pl.DataFrame(
        {
            "num_episodes": num_episodes,
            "num_gains": rng.binomial(num_episodes.astype(int), p),
            "mean_episode_duration": length,
        },
    )


def _pendant_nll(coarsegrain_df, log_gain_rate, rate):
    return tgs_calc_nll(
        gain_rate=np.exp(log_gain_rate),
        growth_rate=rate,
        **make_pendant_nll_kwargs(coarsegrain_df),
    )


def test_survival_discounts():
    data, times = _data()
    n, k, lengths = data["coarsegrain_df"].to_numpy().T
    shares = data["survival_duration_granule_mass"]
    # equivalent to scaling each coarsegrained row's success probability
    rate = -0.05
    factor = shares @ np.exp(rate * times)
    nll, __, __ = tgs_calc_nll(
        gain_rate=1e-3,
        growth_rate=rate,
        **data,
        distinct_survival_durations=times,
    )
    gain = 1e-3 * np.expm1(rate * lengths) / rate * factor
    expected = -np.sum(k * np.log(gain) + (n - k) * np.log1p(-gain))
    assert nll == pytest.approx(expected)


@pytest.mark.parametrize("rate", [-0.1, -1e-9, 0.0, 0.02])
def test_gradient(rate: float):
    data, times = _data()

    def f(params):
        return tgs_calc_nll(
            gain_rate=np.exp(params[0]),
            growth_rate=params[1],
            **data,
            distinct_survival_durations=times,
        )

    x0 = np.array([np.log(1e-3), rate])
    __, dlog, drate = f(x0)
    numeric = optimize.approx_fprime(x0, lambda x: f(x)[0], 1e-7)
    np.testing.assert_allclose([dlog, drate], numeric, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("rate", [-0.1, -1e-8, 0.0, 1e-8, 0.05])
def test_pendant_gradient(rate: float):
    coarsegrain_df = _pendant_coarsegrain_df(1e-3, -0.05, seed=3)

    def nll(params):
        return _pendant_nll(coarsegrain_df, *params)[0]

    params = np.array([np.log(1e-3), rate])
    expected = optimize.approx_fprime(params, nll, [1e-6, 1e-9])
    np.testing.assert_allclose(
        _pendant_nll(coarsegrain_df, *params)[1:], expected, rtol=1e-4
    )


def test_continuous_at_zero_rate():
    coarsegrain_df = _pendant_coarsegrain_df(1e-3, 0.0, seed=4)
    at_zero = _pendant_nll(coarsegrain_df, np.log(1e-3), 0.0)
    near_zero = _pendant_nll(coarsegrain_df, np.log(1e-3), 1e-12)
    np.testing.assert_allclose(at_zero, near_zero, rtol=1e-8)


@pytest.mark.parametrize(
    "rate", [-0.1, -1e-4, -1e-7, -1e-9, 0.0, 1e-9, 1e-7, 1e-4, 0.05]
)
def test_effective_length(rate: float):
    length = np.array([0.01, 1.0, 7.0, 30.0])
    res, res_deriv = _tgs_calc_effective_length(
        length=length, growth_rate=rate
    )
    # exact values, from the series sums over j >= 1 of
    # rate**(j - 1) * t**j / j!, and over j >= 2 of its derivative,
    # (j - 1) * rate**(j - 2) * t**j / j!
    expected = [
        float(
            sum(
                Fraction(rate) ** (j - 1) * Fraction(t) ** j / factorial(j)
                for j in range(1, 80)
            )
        )
        for t in length
    ]
    expected_deriv = [
        float(
            sum(
                (j - 1)
                * Fraction(rate) ** (j - 2)
                * Fraction(t) ** j
                / factorial(j)
                for j in range(2, 80)
            )
        )
        for t in length
    ]
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    np.testing.assert_allclose(res_deriv, expected_deriv, rtol=1e-9)
