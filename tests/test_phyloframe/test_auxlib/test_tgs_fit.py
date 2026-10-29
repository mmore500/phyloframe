import numpy as np
import pytest
from scipy import optimize

from phyloframe._auxlib._tgs_fit import _tgs_calc_nll, tgs_fit


def _sample(gain_rate: float, selection: float, seed: int):
    rng = np.random.default_rng(seed)
    pendant_length = np.linspace(0.5, 60.0, 120)
    num_trials = np.full_like(pendant_length, 20_000)
    p = (
        gain_rate * pendant_length
        if selection == 0
        else gain_rate * np.expm1(selection * pendant_length) / selection
    )
    return num_trials, rng.binomial(num_trials.astype(int), p), pendant_length


def _fit(n, k, x, **kwargs):
    return tgs_fit(num_trials=n, num_successes=k, pendant_length=x, **kwargs)


@pytest.mark.parametrize("gain_rate", [1e-3, 2e-3])
@pytest.mark.parametrize("selection", [0.0, -0.05, -0.1])
def test_recovers_parameters(gain_rate: float, selection: float):
    gain, sel, lrt = _fit(*_sample(gain_rate, selection, seed=1))
    assert gain == pytest.approx(gain_rate, rel=0.1)
    assert sel == pytest.approx(selection, abs=0.01)
    if selection == 0:
        assert lrt < 10
    else:
        assert lrt > 25


def test_loss_rate_shifts_selection():
    n, k, x = _sample(1e-3, -0.05, seed=2)
    __, sel0, __ = _fit(n, k, x)
    __, sel1, __ = _fit(n, k, x, loss_rate=0.01)
    assert sel1 == pytest.approx(sel0 + 0.01, abs=1e-3)


def test_no_successes():
    with pytest.warns(UserWarning, match="no trait carriers"):
        gain, sel, lrt = _fit([10, 10], [0, 0], [1.0, 2.0])
    assert gain == 0.0
    assert np.isnan(sel) and np.isnan(lrt)


def test_empty_bins_ignored():
    n, k, x = _sample(1e-3, -0.05, seed=3)
    expected = _fit(n, k, x)
    res = _fit(np.append(n, 0), np.append(k, 0), np.append(x, 0.0))
    assert res == pytest.approx(expected)


@pytest.mark.parametrize("rate", [-0.1, -1e-8, 0.0, 1e-8, 0.05])
def test_tgs_calc_nll_gradient(rate: float):
    n, k, x = _sample(1e-3, -0.05, seed=3)

    def nll(params):
        return _tgs_calc_nll(
            n, k, x, np.exp(params[0]), params[1], max_exponent=600.0
        )[0]

    def grad(params):
        return _tgs_calc_nll(
            n, k, x, np.exp(params[0]), params[1], max_exponent=600.0
        )[1:]

    params = np.array([np.log(1e-3), rate])
    expected = optimize.approx_fprime(params, nll, [1e-6, 1e-9])
    np.testing.assert_allclose(grad(params), expected, rtol=1e-4)


def test_tgs_calc_nll_continuous_at_zero_rate():
    n, k, x = _sample(1e-3, 0.0, seed=4)
    at_zero = _tgs_calc_nll(n, k, x, 1e-3, 0.0, max_exponent=600.0)
    near_zero = _tgs_calc_nll(n, k, x, 1e-3, 1e-12, max_exponent=600.0)
    np.testing.assert_allclose(at_zero, near_zero, rtol=1e-8)


def test_result_fields():
    res = _fit(*_sample(1e-3, -0.05, seed=5))
    gain, sel, lrt = res
    assert (res.gain_rate, res.selection, res.selection_lrt_naive) == (
        gain,
        sel,
        lrt,
    )


@pytest.mark.parametrize("selection_starts", [(0.0,), (-1.8,), (-0.3, 0.0)])
def test_selection_starts(selection_starts):
    data = _sample(1e-3, -0.05, seed=6)
    expected = _fit(*data)
    res = _fit(*data, selection_starts=selection_starts)
    assert res.selection == pytest.approx(expected.selection, rel=1e-3)
    assert res.gain_rate == pytest.approx(expected.gain_rate, rel=1e-3)


def test_tgs_calc_nll_finite_for_extreme_rate():
    # optimizer trial steps far into positive selection must not overflow
    n, k, x = _sample(1e-3, -0.05, seed=7)
    with np.errstate(over="raise", invalid="raise"):
        nll, dlog_gain, drate = _tgs_calc_nll(
            n, k, x, 1e-3, 1e3, max_exponent=600.0
        )
    assert np.isfinite([nll, dlog_gain, drate]).all()


def test_no_convergence_warns(monkeypatch):
    from scipy import optimize

    def fail(*args, **kwargs):
        res = optimize.minimize(*args, **kwargs)
        res.success, res.message = False, "forced failure"
        return res

    monkeypatch.setattr(
        "phyloframe._auxlib._tgs_fit.optimize",
        type("shim", (), {"minimize": staticmethod(fail)}),
    )
    with pytest.warns(UserWarning, match="did not converge"):
        _fit(*_sample(1e-3, -0.05, seed=8))


def test_keyword_only():
    with pytest.raises(TypeError):
        tgs_fit(*_sample(1e-3, -0.05, seed=9))


@pytest.mark.parametrize(
    "rate", [-0.1, -1e-4, -1e-7, -1e-9, 0.0, 1e-9, 1e-7, 1e-4, 0.05]
)
def test_tgs_calc_effective_length_deriv(rate: float):
    from fractions import Fraction
    from math import factorial

    from phyloframe._auxlib._tgs_fit import _tgs_calc_effective_length_deriv

    pendant_length = np.array([0.01, 1.0, 7.0, 30.0])
    res = _tgs_calc_effective_length_deriv(
        pendant_length=pendant_length, rate=rate, max_exponent=600.0
    )
    # exact derivative, from the series sum over j >= 2 of
    # (j - 1) * rate**(j - 2) * t**j / j!
    expected = [
        float(
            sum(
                (j - 1)
                * Fraction(rate) ** (j - 2)
                * Fraction(t) ** j
                / factorial(j)
                for j in range(2, 80)
            )
        )
        for t in pendant_length
    ]
    np.testing.assert_allclose(res, expected, rtol=1e-9)
