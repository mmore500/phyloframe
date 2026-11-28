import typing

from ._jit import jit


# adapted from https://github.com/mmore500/hstrat/blob/dcf772ef/hstrat/_auxiliary_lib/_reversed_range.py
def reversed_range(start: int, stop: typing.Optional[int] = None) -> range:
    """Create a range in reverse order.

    Equivalent to `reversed(range(*args))`. Compatible with numba jit.
    """
    # numba jit doesn't support reversed
    if stop is None:
        return range(start - 1, -1, -1)
    else:
        return range(stop - 1, start - 1, -1)


reversed_range_jit = jit(nopython=True)(reversed_range)
