import math

import pytest

from phyloframe._auxlib import clamp, jit


@pytest.mark.parametrize(
    "value, lo, hi, expected",
    [
        (5.0, 0.0, 3.0, 3.0),
        (-1.0, 0.0, 3.0, 0.0),
        (2.0, 0.0, 3.0, 2.0),
        (0.0, 0.0, 3.0, 0.0),
        (3.0, 0.0, 3.0, 3.0),
        (1.0, 1.0, 1.0, 1.0),
        (7, 2, 4, 4),
        (-7, -4, -2, -4),
    ],
)
def test_clamp(value, lo, hi, expected):
    assert clamp(value, lo=lo, hi=hi) == expected


def test_clamp_nan():
    # std::clamp semantics: comparisons with nan are false, so nan passes
    assert math.isnan(clamp(math.nan, lo=0.0, hi=1.0))


def test_clamp_within_jit():
    @jit(nopython=True)
    def clamp_sum(lo: float, hi: float) -> float:
        total = 0.0
        for i in range(-5, 6):
            total += clamp(float(i), lo, hi)
        return total

    assert clamp_sum(-1.0, 2.0) == -1.0 * 5 + 0.0 + 1.0 + 2.0 * 4


def test_clamp_invalid_bounds():
    with pytest.raises(AssertionError):
        clamp(0.0, lo=1.0, hi=0.0)


@pytest.mark.parametrize(
    "value, lo, hi, expected",
    [(5, 1, None, 5), (0, 1, None, 1), (9, None, 3, 3), (2, None, None, 2)],
)
def test_clamp_none_bound_unapplied(value, lo, hi, expected):
    res = clamp(value, lo=lo, hi=hi)
    assert res == expected
    assert isinstance(res, int)
