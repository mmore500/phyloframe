import numpy as np
import pytest
from scipy import special, stats

from phyloframe._auxlib._calc_binomial_nll import calc_binomial_nll


def _sample(seed: int):
    rng = np.random.default_rng(seed)
    num_trials = rng.integers(0, 50, 100).astype(float)
    p = rng.uniform(0.01, 0.99, 100)
    return num_trials, rng.binomial(num_trials.astype(int), p), p


@pytest.mark.parametrize("seed", range(3))
def test_matches_scipy_up_to_constant(seed: int):
    n, k, p = _sample(seed)
    nll, __ = calc_binomial_nll(n, k, p)
    log_comb = np.sum(
        special.gammaln(n + 1)
        - special.gammaln(k + 1)
        - special.gammaln(n - k + 1)
    )
    expected = -np.sum(stats.binom.logpmf(k, n, p))
    assert nll - log_comb == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_gradient_matches_finite_difference(seed: int):
    n, k, p = _sample(seed)
    __, grad = calc_binomial_nll(n, k, p)
    h = 1e-7
    for i in range(0, len(p), 10):
        step = np.zeros_like(p)
        step[i] = h
        fd = (
            calc_binomial_nll(n, k, p + step)[0]
            - calc_binomial_nll(n, k, p - step)[0]
        ) / (2 * h)
        assert grad[i] == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_minimized_at_observed_proportion():
    n, k = np.array([10.0, 20.0]), np.array([3.0, 5.0])
    p = k / n
    nll, grad = calc_binomial_nll(n, k, p)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)
    assert calc_binomial_nll(n, k, p + 0.01)[0] > nll
    assert calc_binomial_nll(n, k, p - 0.01)[0] > nll


def test_out_of_range_p_clipped():
    n, k = np.array([10.0, 10.0, 10.0]), np.array([0.0, 10.0, 5.0])
    nll, grad = calc_binomial_nll(n, k, np.array([-0.5, 1.5, 0.0]))
    assert np.isfinite(nll)
    np.testing.assert_array_equal(grad, 0.0)


def test_boundary_p_exact():
    # p of 0 or 1 is exact when consistent with data, giving zero nll
    nll, grad = calc_binomial_nll(
        np.array([10.0, 10.0]), np.array([0.0, 10.0]), np.array([0.0, 1.0])
    )
    assert nll == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(grad, 0.0)


def test_empty():
    nll, grad = calc_binomial_nll(np.array([]), np.array([]), np.array([]))
    assert nll == 0.0
    assert grad.shape == (0,)
