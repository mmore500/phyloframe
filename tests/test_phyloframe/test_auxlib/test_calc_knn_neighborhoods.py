import numpy as np
import pytest

from phyloframe._auxlib._calc_knn_neighborhoods import calc_knn_neighborhoods


def test_line():
    points = np.array([[0.0], [1.0], [2.0], [10.0]])
    res = calc_knn_neighborhoods(points, num_neighbors=2).toarray()
    np.testing.assert_array_equal(
        res,
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    )


def test_includes_self():
    points = np.random.default_rng(1).random((50, 2))
    res = calc_knn_neighborhoods(points, num_neighbors=1)
    np.testing.assert_array_equal(res.toarray(), np.eye(50))


@pytest.mark.parametrize("num_neighbors", [0, 3, 50, 100])
def test_row_sizes(num_neighbors: int):
    points = np.random.default_rng(2).random((50, 2))
    res = calc_knn_neighborhoods(points, num_neighbors=num_neighbors)
    assert res.shape == (50, 50)
    expected = min(max(num_neighbors, 1), 50)
    np.testing.assert_array_equal(res.sum(axis=1).A1, expected)
