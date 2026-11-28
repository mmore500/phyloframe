import typing

from ._jit import jit


# adapted from https://github.com/mmore500/hstrat/blob/dcf772ef/hstrat/_auxiliary_lib/_reversed_range.py
def reversed_range(
    start: int, stop: typing.Optional[int] = None
) -> typing.Iterator[int]:
    """Iterate through a range in reverse order.

    Equivalent to `reversed(range(*args))`. Compatible with numba jit.
    """
    # numba jit doesn't support yield from or reversed
    cur = stop if stop is not None else start
    last = start if stop is not None else 0
    while cur > last:
        cur -= 1
        yield cur


reversed_range_jit = jit(nopython=True)(reversed_range)
