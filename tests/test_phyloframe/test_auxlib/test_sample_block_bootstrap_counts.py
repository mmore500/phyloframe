import numpy as np
import pytest

from phyloframe._auxlib._RngStateContext import RngStateContext
from phyloframe._auxlib._sample_block_bootstrap_counts import (
    sample_block_bootstrap_counts,
)


@pytest.mark.parametrize("num_blocks", [1, 2, 7, 100])
def test_shape_and_row_sums(num_blocks: int):
    counts = sample_block_bootstrap_counts(num_blocks, 20)
    assert counts.shape == (20, num_blocks)
    assert (counts >= 0).all()
    np.testing.assert_array_equal(counts.sum(axis=1), num_blocks)


def test_mean_count_one():
    counts = sample_block_bootstrap_counts(10, 5_000)
    np.testing.assert_allclose(counts.mean(axis=0), 1.0, atol=0.1)


def test_seeded_by_rng_state_context():
    with RngStateContext(1):
        a = sample_block_bootstrap_counts(50, 10)
    with RngStateContext(1):
        b = sample_block_bootstrap_counts(50, 10)
    with RngStateContext(2):
        c = sample_block_bootstrap_counts(50, 10)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_empty():
    assert sample_block_bootstrap_counts(0, 3).shape == (3, 0)
    assert sample_block_bootstrap_counts(5, 0).shape == (0, 5)
