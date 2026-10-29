import numpy as np
import pytest

from phyloframe._auxlib._calc_kdtree_partitions import calc_kdtree_partitions


@pytest.mark.parametrize("num_points", [1_000, 1_001, 4_097])
@pytest.mark.parametrize("num_levels", [1, 3, 6])
def test_equal_occupancy(num_points: int, num_levels: int):
    points = np.random.default_rng(1).random((num_points, 2))
    partitions = calc_kdtree_partitions(points, num_levels)
    counts = np.bincount(partitions, minlength=2**num_levels)
    assert len(counts) == 2**num_levels
    assert counts.max() - counts.min() <= 1


def test_equal_occupancy_correlated():
    # strongly correlated coordinates, as with ranks of pendant length and
    # origin time; per-coordinate quantile grids leave many cells empty
    rng = np.random.default_rng(2)
    x = rng.random(10_000)
    points = np.column_stack([x, x + 0.05 * rng.random(10_000)])
    counts = np.bincount(calc_kdtree_partitions(points, 5), minlength=32)
    assert counts.max() - counts.min() <= 1


def test_partitions_are_spatially_compact():
    points = np.random.default_rng(3).random((4_000, 2))
    partitions = calc_kdtree_partitions(points, 4)
    within = np.mean(
        [points[partitions == p].std(axis=0).mean() for p in range(16)]
    )
    assert within < points.std(axis=0).mean() / 2


def test_zero_levels():
    points = np.random.default_rng(4).random((10, 2))
    np.testing.assert_array_equal(calc_kdtree_partitions(points, 0), 0)


def test_too_few_points_raises():
    with pytest.raises(NotImplementedError):
        calc_kdtree_partitions(np.array([[0.0, 0.0], [1.0, 1.0]]), 4)


def test_empty():
    assert calc_kdtree_partitions(np.empty((0, 2)), 0).shape == (0,)
