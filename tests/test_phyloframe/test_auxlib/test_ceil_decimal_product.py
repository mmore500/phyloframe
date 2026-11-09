import numpy as np
import pytest

from phyloframe._auxlib import ceil_decimal_product


@pytest.mark.parametrize(
    "value, count, expected",
    [
        (0.07, 100, 7),  # 0.07 * 100 == 7.000000000000001 in float
        (0.29, 100, 29),  # 0.29 * 100 == 28.999999999999996 in float
        (0.071, 100, 8),
        (0.1, 30, 3),
        (0.005, 256, 2),
        (1.0, 256, 256),
        (0.0, 10, 0),
        (1e-15, 100, 1),
        (np.float64(0.07), 100, 7),
    ],
)
def test_ceil_decimal_product(value: float, count: int, expected: int):
    result = ceil_decimal_product(value, count)
    assert result == expected
    assert isinstance(result, int)
