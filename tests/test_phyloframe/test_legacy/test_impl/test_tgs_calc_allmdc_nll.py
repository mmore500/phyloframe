import numpy as np
import pytest
from scipy import optimize

from phyloframe._auxlib._tgs_fit import _tgs_calc_nll
from phyloframe.legacy._impl import tgs_calc_allmdc_nll


def _data():
    rng = np.random.default_rng(1)
    n = rng.integers(50, 100, 6).astype(float)
    k = rng.integers(0, 5, 6).astype(float)
    lengths = rng.uniform(1, 20, 6)
    shares = rng.random((6, 3))
    shares /= shares.sum(axis=1, keepdims=True)
    return (n, k, lengths, shares), np.array([0.0, 5.0, 30.0])


def test_matches_pendant_without_survival_time():
    (n, k, lengths, __), __ = _data()
    data = (n, k, lengths, np.ones((6, 1)))
    res = tgs_calc_allmdc_nll(
        log_gain_rate=np.log(1e-3),
        rate=-0.05,
        data=data,
        survival_bin_times=np.array([0.0]),
        max_growth_exponent=600.0,
    )
    expected = _tgs_calc_nll(
        n, k, lengths, 1e-3, -0.05, max_growth_exponent=600.0
    )
    np.testing.assert_allclose(res, expected)


def test_survival_discounts():
    data, times = _data()
    n, k, lengths, shares = data
    # equivalent to scaling each partition's success probability
    rate = -0.05
    factor = shares @ np.exp(rate * times)
    nll, __, __ = tgs_calc_allmdc_nll(
        log_gain_rate=np.log(1e-3),
        rate=rate,
        data=data,
        survival_bin_times=times,
        max_growth_exponent=600.0,
    )
    gain = 1e-3 * np.expm1(rate * lengths) / rate * factor
    expected = -np.sum(k * np.log(gain) + (n - k) * np.log1p(-gain))
    assert nll == pytest.approx(expected)


@pytest.mark.parametrize("rate", [-0.1, -1e-9, 0.0, 0.02])
def test_gradient(rate: float):
    data, times = _data()

    def f(params):
        return tgs_calc_allmdc_nll(
            log_gain_rate=params[0],
            rate=params[1],
            data=data,
            survival_bin_times=times,
            max_growth_exponent=600.0,
        )

    x0 = np.array([np.log(1e-3), rate])
    __, dlog, drate = f(x0)
    numeric = optimize.approx_fprime(x0, lambda x: f(x)[0], 1e-7)
    np.testing.assert_allclose([dlog, drate], numeric, rtol=1e-4, atol=1e-3)
