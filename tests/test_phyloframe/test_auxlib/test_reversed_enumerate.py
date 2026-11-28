import typing

import numpy as np

from phyloframe._auxlib import jit, reversed_enumerate
from phyloframe._auxlib._reversed_enumerate import reversed_enumerate_jit


def test_reversed_enumerate_list():
    lst = [1, 2, 3, 4]
    assert list(reversed_enumerate(lst)) == [(3, 4), (2, 3), (1, 2), (0, 1)]


def test_reversed_enumerate_str():
    s = "hello"
    assert list(reversed_enumerate(s)) == [
        (4, "o"),
        (3, "l"),
        (2, "l"),
        (1, "e"),
        (0, "h"),
    ]


def test_reversed_enumerate_empty():
    e = []
    assert list(reversed_enumerate(e)) == []


def test_reversed_enumerate_single_element():
    e = [1]
    assert list(reversed_enumerate(e)) == [(0, 1)]


def test_reversed_enumerate_tuple():
    t = (1, 2, 3)
    assert list(reversed_enumerate(t)) == [(2, 3), (1, 2), (0, 1)]


def test_reversed_enumerate_is_reverse():
    lst = [1, 2, 3, 4]
    assert [*reversed_enumerate(lst)] == [*reversed([*enumerate(lst)])]


def test_reversed_enumerate_jit():
    a = np.array([1, 2, 3, 4])
    assert list(reversed_enumerate_jit(a)) == [(3, 4), (2, 3), (1, 2), (0, 1)]


@jit(nopython=True)
def _collect_jit(sequence: np.ndarray) -> typing.List[typing.Tuple[int, int]]:
    return [(i, item) for i, item in reversed_enumerate_jit(sequence)]


def test_reversed_enumerate_jit_within_jit():
    a = np.array([1, 2, 3, 4])
    assert _collect_jit(a) == [(3, 4), (2, 3), (1, 2), (0, 1)]
    assert _collect_jit(a[:1]) == [(0, 1)]
    assert _collect_jit(a[:0]) == []
