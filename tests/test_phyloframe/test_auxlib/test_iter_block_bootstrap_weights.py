import numpy as np
import pytest

from phyloframe._auxlib._iter_block_bootstrap_weights import (
    iter_block_bootstrap_weights,
)


def test_weights_constant_within_blocks():
    block_ids = np.array([0, 0, 1, 2, 2, 2, 1])
    for weights in iter_block_bootstrap_weights(block_ids, 20, seed=1):
        assert weights.shape == block_ids.shape
        assert weights.dtype == np.float64
        for block in range(3):
            assert len(set(weights[block_ids == block])) == 1


def test_draws_num_blocks():
    block_ids = np.repeat(np.arange(5), [1, 2, 3, 4, 5])
    for weights in iter_block_bootstrap_weights(block_ids, 20, seed=1):
        block_weights = [weights[block_ids == b][0] for b in range(5)]
        assert sum(block_weights) == 5
        assert all(w >= 0 and w == int(w) for w in block_weights)


def test_mean_weight_one():
    block_ids = np.arange(1_000) % 10
    weights = np.array([*iter_block_bootstrap_weights(block_ids, 2_000, 1)])
    assert weights.mean(axis=0) == pytest.approx(np.ones(1_000), abs=0.1)


def test_seed_reproducible():
    block_ids = np.arange(100) % 7
    a, b, c = (
        np.array([*iter_block_bootstrap_weights(block_ids, 10, seed=s)])
        for s in (1, 1, 2)
    )
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_replicates_independent_of_count():
    # each replicate has its own stream, so prefixes agree
    block_ids = np.arange(100) % 7
    short = [*iter_block_bootstrap_weights(block_ids, 3, seed=1)]
    long = [*iter_block_bootstrap_weights(block_ids, 10, seed=1)]
    np.testing.assert_array_equal(short, long[:3])


def test_empty():
    assert [*iter_block_bootstrap_weights(np.array([0, 1]), 0)] == []
    (weights,) = iter_block_bootstrap_weights(np.array([], dtype=int), 1)
    assert weights.shape == (0,)
