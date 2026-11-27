import typing

from ._jit import jit


@jit(nopython=True)
def cmp(first: typing.Sequence, second: typing.Sequence) -> int:
    """Compare equal-length sequences `first` and `second` by their first
    differing items.

    Returns -1 if `first` orders before `second`, 1 if after, and 0 if all
    items are equal.
    """
    # loop, as numba does not support next over a generator expression
    for a, b in zip(first, second):
        if a != b:
            return int(a > b) - int(a < b)
    return 0
