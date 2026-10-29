import numpy as np
from scipy import sparse


def calc_neighborhood_rate_ratios(
    *,
    neighborhoods: sparse.spmatrix,
    num_events: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    """Calculate, for each item, its neighborhood's event rate relative to
    the overall event rate.

    Each item's ratio is events observed within its neighborhood divided by
    events expected there at the overall rate, i.e., a local standardized
    rate ratio. Exposure is the opportunity for events to occur (e.g.,
    summed branch length, analogous to person-time).

    Parameters
    ----------
    neighborhoods : scipy.sparse matrix
        Square membership matrix, whose row `i` marks (with 1) the items in
        item `i`'s neighborhood, e.g., from
        `sklearn.neighbors.kneighbors_graph`.
    num_events : np.ndarray
        Event count of each item.
    exposure : np.ndarray
        Exposure of each item.

    Returns
    -------
    np.ndarray
        Rate ratio of each item's neighborhood; 1.0 where a neighborhood has
        no exposure, or where there are no events at all.
    """
    rate = num_events.sum() / max(exposure.sum(), np.finfo(float).tiny)
    observed = neighborhoods @ num_events
    expected = rate * (neighborhoods @ exposure)
    return np.divide(
        observed, expected, out=np.ones_like(observed), where=expected > 0
    )
