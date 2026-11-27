import numpy as np
import pytest

from phyloframe._auxlib import cmplex


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ((), (), 0),
        ((1,), (1,), 0),
        ((1,), (2,), -1),
        ((2,), (1,), 1),
        ((1, 2, 3), (1, 2, 3), 0),
        ((1, 2, 3), (1, 2, 4), -1),
        ((1, 3, 0), (1, 2, 9), 1),  # first differing items decide
        ((0, 9, 9), (1, 0, 0), -1),
        ((1.5, 2.0), (1.5, -2.0), 1),
    ],
)
def test_cmplex(first: tuple, second: tuple, expected: int):
    assert cmplex(np.array(first), np.array(second)) == expected
    assert cmplex(np.array(second), np.array(first)) == -expected
    if first:
        assert cmplex(first, second) == expected


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64])
def test_cmplex_unsigned(dtype: type):
    first = np.array([1, 5], dtype=dtype)
    second = np.array([1, 7], dtype=dtype)
    assert cmplex(first, second) == -1
    assert cmplex(second, first) == 1
    assert cmplex(first, first) == 0


def test_cmplex_rows():
    values = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]])
    assert cmplex(values[0], values[1]) == 0
    assert cmplex(values[0], values[2]) == -1
    assert cmplex(values[2], values[1]) == 1
