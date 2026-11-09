import numpy as np
import pytest

from phyloframe._auxlib import calc_tolerant_ceil


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.07 * 100, 7),  # 7.000000000000001 in float
        (0.1 * 3 * 10, 3),  # 3.0000000000000004 in float
        (0.29 * 100, 29),  # 28.999999999999996 in float
        (0.071 * 100, 8),
        (1e-15 * 100, 1),
        (256.0, 256),
        (1.28, 2),
        (0.0, 0),
        (-2.5, -2),
        (-0.07 * 100, -7),
    ],
)
def test_scalar(value: float, expected: int):
    assert calc_tolerant_ceil(value) == expected


def test_array():
    values = np.array([0.07, 0.071, 0.005]) * 100
    np.testing.assert_array_equal(calc_tolerant_ceil(values), [7, 8, 1])
    assert calc_tolerant_ceil(values).dtype == np.int64


def test_empty():
    assert calc_tolerant_ceil(np.zeros(0)).shape == (0,)


def test_rtol():
    assert calc_tolerant_ceil(7.001) == 8
    assert calc_tolerant_ceil(7.001, rtol=1e-3) == 7
