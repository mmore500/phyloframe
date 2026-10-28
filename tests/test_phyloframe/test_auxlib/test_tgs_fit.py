import numpy as np
import pytest
from scipy import optimize

from phyloframe._auxlib._tgs_fit import _tgs_calc_nll, tgs_fit


def _sample(gain_rate: float, selection: float, seed: int):
    rng = np.random.default_rng(seed)
    exposure = np.linspace(0.5, 60.0, 120)
    num_trials = np.full_like(exposure, 20_000)
    p = (
        gain_rate * exposure
        if selection == 0
        else gain_rate * np.expm1(selection * exposure) / selection
    )
    return num_trials, rng.binomial(num_trials.astype(int), p), exposure


@pytest.mark.parametrize("gain_rate", [1e-3, 2e-3])
@pytest.mark.parametrize("selection", [0.0, -0.05, -0.1])
def test_recovers_parameters(gain_rate: float, selection: float):
    gain, sel, lrt = tgs_fit(
        *_sample(gain_rate, selection, seed=1),
    )
    assert gain == pytest.approx(gain_rate, rel=0.1)
    assert sel == pytest.approx(selection, abs=0.01)
    if selection == 0:
        assert lrt < 10
    else:
        assert lrt > 25


def test_loss_rate_shifts_selection():
    n, k, x = _sample(1e-3, -0.05, seed=2)
    __, sel0, __ = tgs_fit(n, k, x)
    __, sel1, __ = tgs_fit(n, k, x, loss_rate=0.01)
    assert sel1 == pytest.approx(sel0 + 0.01, abs=1e-3)


def test_no_successes():
    gain, sel, lrt = tgs_fit([10, 10], [0, 0], [1.0, 2.0])
    assert gain == 0.0
    assert np.isnan(sel) and np.isnan(lrt)


def test_empty_bins_ignored():
    n, k, x = _sample(1e-3, -0.05, seed=3)
    expected = tgs_fit(n, k, x)
    res = tgs_fit(np.append(n, 0), np.append(k, 0), np.append(x, 0.0))
    assert res == pytest.approx(expected)


@pytest.mark.parametrize("rate", [-0.1, -1e-8, 0.0, 1e-8, 0.05])
def test_tgs_calc_nll_gradient(rate: float):
    n, k, x = _sample(1e-3, -0.05, seed=3)

    def nll(params):
        return _tgs_calc_nll(n, k, x, np.exp(params[0]), params[1])[0]

    def grad(params):
        return _tgs_calc_nll(n, k, x, np.exp(params[0]), params[1])[1:]

    params = np.array([np.log(1e-3), rate])
    expected = optimize.approx_fprime(params, nll, [1e-6, 1e-9])
    np.testing.assert_allclose(grad(params), expected, rtol=1e-4)


def test_tgs_calc_nll_continuous_at_zero_rate():
    n, k, x = _sample(1e-3, 0.0, seed=4)
    at_zero = _tgs_calc_nll(n, k, x, 1e-3, 0.0)
    near_zero = _tgs_calc_nll(n, k, x, 1e-3, 1e-12)
    np.testing.assert_allclose(at_zero, near_zero, rtol=1e-8)
