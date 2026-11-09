import numpy as np


def calc_geometric_bins(values: np.ndarray, *, ratio: float) -> np.ndarray:
    """Assign each value to a bin whose bounds grow geometrically by `ratio`.

    Bin `k` holds values in (ratio ** (k - 1), ratio ** k], so, e.g., ratio
    2 gives bins of 1, 2, 3-4, 5-8, 9-16, ... for integer values.

    Parameters
    ----------
    values : np.ndarray
        Values to bin, each positive.
    ratio : float
        Ratio between successive bin bounds; must exceed 1.

    Returns
    -------
    np.ndarray
        Integer bin of each value.
    """
    values = np.asarray(values)
    assert ratio > 1
    assert (values > 0).all()
    k = np.ceil(np.log(values) / np.log(ratio)).astype(np.int64)
    # correct floating point error at bin bounds
    k -= np.float_power(ratio, k - 1) >= values
    k += np.float_power(ratio, k) < values
    return k
