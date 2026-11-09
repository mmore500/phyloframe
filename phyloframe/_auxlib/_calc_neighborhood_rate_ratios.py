import warnings

import numpy as np
from scipy import sparse


def calc_neighborhood_rate_ratios(
    *,
    neighborhoods: sparse.spmatrix,
    num_events: np.ndarray,
    exposure: np.ndarray,
    max_ratio: float = np.inf,
    pseudocount: float = 0.0,
) -> np.ndarray:
    """Calculate, for each item, its neighborhood's event rate relative to
    the global event rate.

    Each item's ratio is events observed within its neighborhood divided by
    events expected there at the global rate, i.e., a local standardized
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
    max_ratio : float, default np.inf
        Cap on each rate ratio, e.g., to bound ratios of neighborhoods with
        little exposure.
    pseudocount : float, default 0.0
        Count added to both observed and expected events, shrinking each
        ratio toward 1 and keeping it nonzero where a neighborhood has no
        events.

    Returns
    -------
    np.ndarray
        Rate ratio of each item's neighborhood; 1.0 where a neighborhood has
        no exposure, or where there are no events at all.
    """
    total_exposure = exposure.sum()
    if total_exposure <= 0:
        warnings.warn(
            "no exposure, so neighborhood rate ratios default to 1.0",
            stacklevel=2,
        )
        return np.ones(neighborhoods.shape[0])
    global_rate = num_events.sum() / total_exposure  # events per exposure
    observed = neighborhoods @ num_events
    expected = global_rate * (neighborhoods @ exposure)
    ratios = np.divide(
        observed + pseudocount,
        expected + pseudocount,
        out=np.ones_like(observed),
        where=expected + pseudocount > 0,
    )
    return np.clip(ratios, None, max_ratio)
