import numpy as np
import pytest

from phyloframe._auxlib._calc_kdtree_partitions import calc_kdtree_partitions


@pytest.mark.parametrize("num_points", [1_000, 1_001, 4_097])
@pytest.mark.parametrize("leafsize", [16, 100, 500])
def test_partition_sizes(num_points: int, leafsize: int):
    points = np.random.default_rng(1).random((num_points, 2))
    partitions = calc_kdtree_partitions(points, leafsize)
    counts = np.bincount(partitions)
    assert (counts > 0).all()
    assert counts.max() <= leafsize
    assert counts.min() >= leafsize // 2


def test_correlated_coordinates():
    # strongly correlated coordinates, as with ranks of pendant length and
    # origin time; per-coordinate quantile grids leave many cells empty
    rng = np.random.default_rng(2)
    x = rng.random(10_000)
    points = np.column_stack([x, x + 0.05 * rng.random(10_000)])
    counts = np.bincount(calc_kdtree_partitions(points, 300))
    assert counts.max() <= 300
    assert counts.min() >= 150


def test_partitions_are_spatially_compact():
    points = np.random.default_rng(3).random((4_000, 2))
    partitions = calc_kdtree_partitions(points, 250)
    within = np.mean(
        [
            points[partitions == p].std(axis=0).mean()
            for p in range(partitions.max() + 1)
        ]
    )
    assert within < points.std(axis=0).mean() / 2


def test_fewer_points_than_leafsize():
    points = np.random.default_rng(4).random((10, 2))
    np.testing.assert_array_equal(calc_kdtree_partitions(points, 100), 0)


def test_empty():
    assert calc_kdtree_partitions(np.empty((0, 2)), 10).shape == (0,)
