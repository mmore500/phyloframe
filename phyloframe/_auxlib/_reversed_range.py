from ._jit import jit


@jit(nopython=True)
def reversed_range(n: int) -> range:
    """Range over `n` indices in reverse, from `n - 1` down to 0.

    Substitutes for `reversed(range(n))` within jit-compiled code, where
    numba does not support `reversed`.
    """
    return range(n - 1, -1, -1)
