import typing

from ._cmp import cmp
from ._jit import jit


@jit(nopython=True)
def cmpseq(first: typing.Sequence, second: typing.Sequence) -> int:
    """Compare equal-length sequences `first` and `second` by their first
    differing items.

    Returns -1 if `first` orders before `second`, 1 if after, and 0 if all
    items are equal.
    """
    # loop, as numba does not support next over a generator expression
    for a, b in zip(first, second):
        if a != b:
            return cmp(a, b)
    return 0
