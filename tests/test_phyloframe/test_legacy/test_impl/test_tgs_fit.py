import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy._impl._tgs_fit import tgs_fit


def _sample(gain_rate: float, selection_coef: float, seed: int):
    rng = np.random.default_rng(seed)
    pendant_length = np.linspace(0.5, 60.0, 120)
    num_trials = np.full_like(pendant_length, 20_000)
    p = (
        gain_rate * pendant_length
        if selection_coef == 0
        else gain_rate
        * np.expm1(selection_coef * pendant_length)
        / selection_coef
    )
    return num_trials, rng.binomial(num_trials.astype(int), p), pendant_length


def _fit(n, k, x, **kwargs):
    return tgs_fit(num_trials=n, num_successes=k, pendant_length=x, **kwargs)


@pytest.mark.parametrize("gain_rate", [1e-3, 2e-3])
@pytest.mark.parametrize("selection_coef", [0.0, -0.05, -0.1])
def test_recovers_parameters(gain_rate: float, selection_coef: float):
    gain, sel, lrt = _fit(*_sample(gain_rate, selection_coef, seed=1))
    assert gain == pytest.approx(gain_rate, rel=0.1)
    assert sel == pytest.approx(selection_coef, abs=0.01)
    if selection_coef == 0:
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


def test_result_fields():
    res = _fit(*_sample(1e-3, -0.05, seed=5))
    gain, sel, lrt = res
    assert (
        res.gain_rate,
        res.selection_coef,
        res.selection_coef_lrt_naive,
    ) == (
        gain,
        sel,
        lrt,
    )


@pytest.mark.parametrize(
    "selection_coef_starts", [(0.0,), (-1.8,), (-0.3, 0.0)]
)
def test_selection_starts(selection_coef_starts):
    data = _sample(1e-3, -0.05, seed=6)
    expected = _fit(*data)
    res = _fit(*data, selection_coef_starts=selection_coef_starts)
    assert res.selection_coef == pytest.approx(
        expected.selection_coef, rel=1e-3
    )
    assert res.gain_rate == pytest.approx(expected.gain_rate, rel=1e-3)


def test_no_convergence_warns(monkeypatch):
    from scipy import optimize

    def fail(*args, **kwargs):
        res = optimize.minimize(*args, **kwargs)
        res.success, res.message = False, "forced failure"
        return res

    monkeypatch.setattr(
        "phyloframe.legacy._impl._tgs_fit.optimize",
        type("shim", (), {"minimize": staticmethod(fail)}),
    )
    with pytest.warns(UserWarning, match="did not converge"):
        _fit(*_sample(1e-3, -0.05, seed=8))


def test_keyword_only():
    with pytest.raises(TypeError):
        tgs_fit(*_sample(1e-3, -0.05, seed=9))


def test_no_successes_skips_fit_without_warning():
    import warnings

    from phyloframe.legacy._impl import tgs_fit_partitions

    partitions = pd.DataFrame(
        {"num_trials": [10.0], "num_successes": [0.0], "length": [1.0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = tgs_fit_partitions(
            calc_nll=None,
            partitions=partitions,
            loss_rate=0.0,
            selection_coef_starts=(0.0,),
        )
    assert res.gain_rate == 0.0
    assert np.isnan(res.selection_coef) and np.isnan(
        res.selection_coef_lrt_naive
    )
