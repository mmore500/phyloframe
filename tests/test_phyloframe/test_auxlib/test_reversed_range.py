import typing

from phyloframe._auxlib import jit, reversed_range
from phyloframe._auxlib._reversed_range import reversed_range_jit


@jit(nopython=True)
def _collect_jit(start: int) -> typing.List[int]:
    return [i for i in reversed_range_jit(start)]


@jit(nopython=True)
def _collect_jit_stop(start: int, stop: int) -> typing.List[int]:
    return [i for i in reversed_range_jit(start, stop)]


def test_reversed_range_jit():
    # jit-compiled ranges can only be iterated within jit-compiled code
    # Test with stop value specified
    assert _collect_jit_stop(1, 5) == [*reversed(range(1, 5))]
    assert _collect_jit_stop(0, 10) == [*reversed(range(0, 10))]
    assert _collect_jit_stop(-5, -1) == [*reversed(range(-5, -1))]
    assert _collect_jit_stop(5, 5) == [*reversed(range(5, 5))]

    # Test without stop value specified
    assert _collect_jit(5) == [*reversed(range(5))]
    assert _collect_jit(0) == [*reversed(range(0))]


def test_reversed_range_stop_specified():
    # Test with stop value specified
    result = list(reversed_range(1, 5))
    assert result == [*reversed(range(1, 5))]
    result = list(reversed_range(0, 10))
    assert result == [*reversed(range(0, 10))]


def test_reversed_range_no_stop_specified():
    # Test without stop value specified
    result = list(reversed_range(5))
    assert result == [*reversed(range(5))]
    result = list(reversed_range(0))
    assert result == [*reversed(range(0))]


def test_reversed_range_single_element():
    # Test with single-element range
    result = list(reversed_range(5, 6))
    assert result == [*reversed(range(5, 6))]
    result = list(reversed_range(5, 5))
    assert result == [*reversed(range(5, 5))]


def test_reversed_range_negative_values():
    # Test with negative start and stop values
    result = list(reversed_range(-5, -1))
    assert result == [*reversed(range(-5, -1))]


def test_reversed_range_empty_range():
    # Test with an empty range
    result = list(reversed_range(5, 5))
    assert result == [*reversed(range(5, 5))]
    result = list(reversed_range(5, 6))
    assert result == [*reversed(range(5, 6))]
    result = list(reversed_range(0, 0))
    assert result == [*reversed(range(0, 0))]
