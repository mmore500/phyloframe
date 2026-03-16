import numpy as np


def bit_length_numpy(arr: np.ndarray) -> np.ndarray:
    """Vectorized bit_length for a positive integer array.

    Equivalent to applying ``int.bit_length`` element-wise, but operates
    on numpy arrays without a Python loop.

    Splits into 32-bit halves so each half fits exactly in float64,
    making the result correct for all int64 values.

    Parameters
    ----------
    arr : np.ndarray
        1-D array of non-negative integers.

    Returns
    -------
    np.ndarray
        Array of the same length with bit-length of each element.
    """
    _, high_exp = np.frexp(arr >> 32)
    _, low_exp = np.frexp(arr & 0xFFFFFFFF)
    return np.where(high_exp, high_exp + 32, low_exp)
