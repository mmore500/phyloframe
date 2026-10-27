import numpy as np
import pytest

from phyloframe._auxlib._fit_gain_selection_exposure import (
    fit_gain_selection_exposure,
)


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
    gain, sel, lrt = fit_gain_selection_exposure(
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
    __, sel0, __ = fit_gain_selection_exposure(n, k, x)
    __, sel1, __ = fit_gain_selection_exposure(n, k, x, loss_rate=0.01)
    assert sel1 == pytest.approx(sel0 + 0.01, abs=1e-3)


def test_no_successes():
    gain, sel, lrt = fit_gain_selection_exposure([10, 10], [0, 0], [1.0, 2.0])
    assert gain == 0.0
    assert np.isnan(sel) and np.isnan(lrt)


def test_empty_bins_ignored():
    n, k, x = _sample(1e-3, -0.05, seed=3)
    expected = fit_gain_selection_exposure(n, k, x)
    res = fit_gain_selection_exposure(
        np.append(n, 0), np.append(k, 0), np.append(x, 0.0)
    )
    assert res == pytest.approx(expected)
