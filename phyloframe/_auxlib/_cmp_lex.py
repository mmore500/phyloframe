import typing

from ._cmp import cmp
from ._jit import jit


@jit(nopython=True)
def cmp_lex(first: typing.Sequence, second: typing.Sequence) -> int:
    """Compare equal-length sequences `first` and `second` by their first
    differing items.

    Returns -1 if `first` orders before `second`, 1 if after, and 0 if all
    items are equal.
    """
    for a, b in zip(first, second):
        if a != b:
            return cmp(a, b)
    return 0
