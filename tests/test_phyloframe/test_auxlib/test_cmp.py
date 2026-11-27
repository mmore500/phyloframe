import numpy as np
import pytest

from phyloframe._auxlib import cmp


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (1, 1, 0),
        (1, 2, -1),
        (2, 1, 1),
        (-1.5, -1.5, 0),
        (-1.5, 2.0, -1),
        (2.0, -1.5, 1),
        (1, 1.5, -1),
        (False, True, -1),
    ],
)
def test_cmp(first, second, expected: int):
    assert cmp(first, second) == expected
    assert cmp(second, first) == -expected


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64])
def test_cmp_unsigned(dtype: type):
    assert cmp(dtype(5), dtype(7)) == -1
    assert cmp(dtype(7), dtype(5)) == 1
    assert cmp(dtype(5), dtype(5)) == 0


def test_cmp_nan():
    # comparisons with nan are false
    assert cmp(float("nan"), 1.0) == 0
    assert cmp(1.0, float("nan")) == 0
