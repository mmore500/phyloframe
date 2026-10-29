import numpy as np
import pytest

from phyloframe._auxlib._sample_bootstrap_counts import (
    sample_bootstrap_counts,
)


@pytest.mark.parametrize("num_items", [1, 2, 7, 100])
def test_shape_and_sum(num_items: int):
    counts = sample_bootstrap_counts(num_items, seed=1)
    assert counts.shape == (num_items,)
    assert (counts >= 0).all()
    assert counts.sum() == num_items


def test_mean_count_one():
    counts = np.array(
        [sample_bootstrap_counts(10, seed=s) for s in range(5_000)]
    )
    np.testing.assert_allclose(counts.mean(axis=0), 1.0, atol=0.1)


def test_seeded():
    a = sample_bootstrap_counts(50, seed=1)
    b = sample_bootstrap_counts(50, seed=1)
    c = sample_bootstrap_counts(50, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_independent_of_global_state():
    np.random.seed(0)
    a = sample_bootstrap_counts(50, seed=3)
    np.random.seed(1)
    b = sample_bootstrap_counts(50, seed=3)
    np.testing.assert_array_equal(a, b)


def test_empty():
    assert sample_bootstrap_counts(0, seed=1).shape == (0,)
