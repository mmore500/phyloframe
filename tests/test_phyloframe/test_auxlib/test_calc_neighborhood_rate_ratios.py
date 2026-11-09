import numpy as np
import pytest
from scipy import sparse

from phyloframe._auxlib._calc_neighborhood_rate_ratios import (
    calc_neighborhood_rate_ratios,
)


def test_uniform():
    # events proportional to exposure everywhere, so ratios of one
    exposure = np.array([1.0, 2.0, 3.0, 4.0])
    res = calc_neighborhood_rate_ratios(
        neighborhoods=sparse.identity(4, format="csr"),
        num_events=0.5 * exposure,
        exposure=exposure,
    )
    np.testing.assert_allclose(res, 1.0)


def test_local():
    # items 0-1 accrue events at twice the rate of items 2-3
    exposure = np.ones(4)
    num_events = np.array([2.0, 2.0, 1.0, 1.0])
    pairs = sparse.csr_matrix(
        np.kron(np.eye(2), np.ones((2, 2)))  # neighborhoods {0, 1}, {2, 3}
    )
    res = calc_neighborhood_rate_ratios(
        neighborhoods=pairs, num_events=num_events, exposure=exposure
    )
    np.testing.assert_allclose(res, [4 / 3, 4 / 3, 2 / 3, 2 / 3])
    # one global neighborhood averages away local differences
    res = calc_neighborhood_rate_ratios(
        neighborhoods=sparse.csr_matrix(np.ones((4, 4))),
        num_events=num_events,
        exposure=exposure,
    )
    np.testing.assert_allclose(res, 1.0)


def test_no_events():
    res = calc_neighborhood_rate_ratios(
        neighborhoods=sparse.identity(3, format="csr"),
        num_events=np.zeros(3),
        exposure=np.ones(3),
    )
    np.testing.assert_allclose(res, 1.0)


def test_no_exposure_neighborhood():
    res = calc_neighborhood_rate_ratios(
        neighborhoods=sparse.identity(2, format="csr"),
        num_events=np.array([1.0, 0.0]),
        exposure=np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(res, [1.0, 1.0])


def test_no_exposure_warns():
    with pytest.warns(UserWarning, match="no exposure"):
        res = calc_neighborhood_rate_ratios(
            neighborhoods=sparse.identity(2, format="csr"),
            num_events=np.zeros(2),
            exposure=np.zeros(2),
        )
    np.testing.assert_allclose(res, [1.0, 1.0])


def test_max_ratio_caps():
    exposure = np.ones(4)
    num_events = np.array([2.0, 2.0, 1.0, 1.0])
    pairs = sparse.csr_matrix(np.kron(np.eye(2), np.ones((2, 2))))
    res = calc_neighborhood_rate_ratios(
        neighborhoods=pairs,
        num_events=num_events,
        exposure=exposure,
        max_ratio=1.2,
    )
    # caps the high ratios (4 / 3), leaving the low ones (2 / 3)
    np.testing.assert_allclose(res, [1.2, 1.2, 2 / 3, 2 / 3])


def test_pseudocount_keeps_zero_events_finite():
    # item 0 observes no events of 10 expected at the global rate
    res = calc_neighborhood_rate_ratios(
        neighborhoods=sparse.identity(2, format="csr"),
        num_events=np.array([0.0, 20.0]),
        exposure=np.array([1.0, 1.0]),
        pseudocount=1.0,
    )
    assert res[0] == pytest.approx(1 / 11)
