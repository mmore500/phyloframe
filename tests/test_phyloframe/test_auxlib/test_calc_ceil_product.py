import pytest

from phyloframe._auxlib import calc_ceil_product


@pytest.mark.parametrize(
    "fraction, count, expected",
    [
        (0.07, 100, 7),  # 0.07 * 100 == 7.000000000000001 in float
        (0.1, 30, 3),
        (0.071, 100, 8),
        (1e-15, 100, 1),
        (1.0, 256, 256),
        (0.005, 256, 2),
        (0.0, 10, 0),
    ],
)
def test_calc_ceil_product(fraction: float, count: int, expected: int):
    assert calc_ceil_product(fraction, count) == expected
