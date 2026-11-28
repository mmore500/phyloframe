import typing

from ._jit import jit

_T = typing.TypeVar("_T")


@jit(nopython=True)
def cmp(first: _T, second: _T) -> int:
    """Compare values `first` and `second`.

    Returns -1 if `first` orders before `second`, 1 if after, and 0
    otherwise.
    """
    return int(first > second) - int(first < second)
