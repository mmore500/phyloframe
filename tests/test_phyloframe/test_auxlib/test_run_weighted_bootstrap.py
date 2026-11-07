import functools

import numpy as np
import pytest

from phyloframe._auxlib import run_weighted_bootstrap


def _weighted_mean(values: np.ndarray, *, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def test_weights_are_draw_counts():
    res = run_weighted_bootstrap(
        lambda weights: weights, num_items=10, num_replicates=5, seed=1
    )
    assert len(res) == 5
    for weights in res:
        assert weights.dtype == np.float64
        assert weights.sum() == 10
        assert (weights >= 0).all()


def test_reproducible():
    func = functools.partial(_weighted_mean, np.arange(100.0))
    kwargs = dict(num_items=100, num_replicates=4)
    res = run_weighted_bootstrap(func, seed=1, **kwargs)
    assert res == run_weighted_bootstrap(func, seed=1, **kwargs)
    assert res != run_weighted_bootstrap(func, seed=2, **kwargs)


@pytest.mark.parametrize("parallel_backend", ["threading", "loky"])
def test_parallel_matches_serial(parallel_backend: str):
    func = functools.partial(_weighted_mean, np.arange(100.0))
    kwargs = dict(num_items=100, num_replicates=4, seed=1)
    assert run_weighted_bootstrap(
        func, parallel_backend=parallel_backend, **kwargs
    ) == run_weighted_bootstrap(func, **kwargs)
