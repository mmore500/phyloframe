import typing

import pytest

from phyloframe._auxlib import jit, reversed_range


@jit(nopython=True)
def _collect(n: int) -> typing.List[int]:
    return [i for i in reversed_range(n)]


@pytest.mark.parametrize("n", [0, 1, 2, 7])
def test_reversed_range(n: int):
    assert _collect(n) == list(reversed(range(n)))
