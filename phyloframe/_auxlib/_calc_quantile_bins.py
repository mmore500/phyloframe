import typing

import numpy as np


def calc_quantile_bins(
    values: np.ndarray, *, max_bins: int
) -> typing.Tuple[np.ndarray, int]:
    """Assign values to up to `max_bins` bins of near-equal occupancy, with
    bounds at quantiles of `values`.

    Bins whose bounding quantiles tie (e.g., from many equal values) merge,
    so there may be fewer than `max_bins` bins, but every value falls in
    one.

    Parameters
    ----------
    values : np.ndarray
        Values to bin.
    max_bins : int
        Largest number of bins.

    Returns
    -------
    bins : np.ndarray
        Integer bin of each value, ordered by value.
    num_bins : int
        Number of bins, at least 1.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.zeros(0, dtype=np.int64), 1
    edges = np.unique(np.quantile(values, np.linspace(0, 1, max_bins + 1)))
    num_bins = max(len(edges) - 1, 1)
    # inner edges only, so the extremes fall in the first and last bins
    return np.searchsorted(edges[1:-1], values, side="right"), num_bins
