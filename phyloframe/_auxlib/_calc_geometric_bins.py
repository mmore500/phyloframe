import numpy as np


def calc_geometric_bins(values: np.ndarray, *, ratio: float) -> np.ndarray:
    """Assign values of at least 1 to bins on a log scale.

    Bin `k` holds values in (ratio ** (k - 1), ratio ** k], so, e.g., ratio
    2 gives bins of 1, 2, 3-4, 5-8, 9-16, ... for integer values. Unlike
    `ceil(log(value) / log(ratio))`, exact powers of `ratio` stay in the
    bin they close despite floating point error.

    Parameters
    ----------
    values : np.ndarray
        Values to bin, each at least 1.
    ratio : float
        Ratio between successive bin bounds; must exceed 1.

    Returns
    -------
    np.ndarray
        Integer bin of each value.
    """
    values = np.asarray(values)
    if not len(values):
        return np.zeros(0, dtype=np.int64)
    bounds = ratio ** np.arange(np.log(values.max()) // np.log(ratio) + 2)
    return np.searchsorted(bounds, values)
