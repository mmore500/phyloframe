import pytest

from phyloframe._auxlib import ceil_div


@pytest.mark.parametrize(
    "numerator, denominator, expected",
    [(0, 4, 0), (1, 4, 1), (4, 4, 1), (5, 4, 2), (8, 4, 2), (9, 4, 3)],
)
def test_ceil_div(numerator: int, denominator: int, expected: int):
    assert ceil_div(numerator, denominator) == expected


def test_ceil_div_exact_for_large_ints():
    assert ceil_div(10**20 + 1, 10**20) == 2
