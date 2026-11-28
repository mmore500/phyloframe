import numpy as np
import pytest

from phyloframe._auxlib import cmp_lex


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
def test_cmp_lex(first: tuple, second: tuple, expected: int):
    assert cmp_lex(np.array(first), np.array(second)) == expected
    assert cmp_lex(np.array(second), np.array(first)) == -expected
    if first:
        assert cmp_lex(first, second) == expected


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64])
def test_cmp_lex_unsigned(dtype: type):
    first = np.array([1, 5], dtype=dtype)
    second = np.array([1, 7], dtype=dtype)
    assert cmp_lex(first, second) == -1
    assert cmp_lex(second, first) == 1
    assert cmp_lex(first, first) == 0


def test_cmp_lex_rows():
    values = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]])
    assert cmp_lex(values[0], values[1]) == 0
    assert cmp_lex(values[0], values[2]) == -1
    assert cmp_lex(values[2], values[1]) == 1
