import numpy as np


def calc_tolerant_ceil(
    values: np.ndarray, *, rtol: float = 1e-12
) -> np.ndarray:
    """Round each value up to an integer, ignoring floating point error.

    A value within relative tolerance `rtol` of an integer gives that
    integer, so, e.g., 0.07 * 100, which is 7.000000000000001 in floating
    point, gives 7, not 8.

    Parameters
    ----------
    values : np.ndarray
        Values to round, each finite.
    rtol : float, default 1e-12
        Relative tolerance within which a value counts as an integer.

    Returns
    -------
    np.ndarray
        Integer ceiling of each value.
    """
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    nearest = np.round(values)
    is_integer = np.isclose(values, nearest, rtol=rtol, atol=0)
    return np.where(is_integer, nearest, np.ceil(values)).astype(np.int64)
