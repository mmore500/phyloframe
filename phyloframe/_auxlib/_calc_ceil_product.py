import numpy as np


def calc_ceil_product(fraction: float, count: int) -> int:
    """Calculate ceil(`fraction` * `count`), ignoring floating point error
    in the product, e.g., giving 7, not 8, for 0.07 * 100.

    Parameters
    ----------
    fraction : float
        Nonnegative multiplier.
    count : int
        Nonnegative count.

    Returns
    -------
    int
        The ceiling of the product, rounded down where the product is within
        floating point error of the integer below.
    """
    assert fraction >= 0
    assert count >= 0
    product = fraction * count
    ceil = int(np.ceil(product))
    return ceil - int(np.isclose(ceil - 1, product, rtol=1e-12, atol=0))
