import numpy as np
import pytest

from phyloframe._auxlib._sample_bootstrap_counts import (
    sample_bootstrap_counts,
)


@pytest.mark.parametrize("num_items", [1, 2, 7, 100])
def test_shape_and_sum(num_items: int):
    counts = sample_bootstrap_counts(num_items, rng=np.random.default_rng(1))
    assert counts.shape == (num_items,)
    assert (counts >= 0).all()
    assert counts.sum() == num_items


def test_mean_count_one():
    rng = np.random.default_rng(1)
    counts = np.array(
        [sample_bootstrap_counts(10, rng=rng) for __ in range(5_000)]
    )
    np.testing.assert_allclose(counts.mean(axis=0), 1.0, atol=0.1)


def test_seeded_by_rng():
    a = sample_bootstrap_counts(50, rng=np.random.default_rng(1))
    b = sample_bootstrap_counts(50, rng=np.random.default_rng(1))
    c = sample_bootstrap_counts(50, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_does_not_use_global_state():
    np.random.seed(1)
    expected = np.random.random()
    np.random.seed(1)
    sample_bootstrap_counts(50, rng=np.random.default_rng(2))
    assert np.random.random() == expected


def test_empty():
    rng = np.random.default_rng(1)
    assert sample_bootstrap_counts(0, rng=rng).shape == (0,)
