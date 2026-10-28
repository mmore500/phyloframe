import numpy as np


def estimate_standardized_rate_ratios(
    *,
    num_events: np.ndarray,
    exposure: np.ndarray,
    group_ids: np.ndarray,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Estimate each group's event rate relative to the overall event rate,
    smoothed toward 1.

    Exposure is the opportunity for events to occur (e.g., summed branch
    length, analogous to person-time), so a group's event rate is its events
    per unit exposure. The ratio of a group's observed events to the events
    expected at the overall rate is its standardized rate ratio (as with
    standardized incidence ratios in epidemiology). To avoid zero or
    unbounded ratios in sparse groups, `pseudocount` expected events at the
    overall rate are added to both observed and expected counts, i.e., a
    gamma prior with mean 1 (Clayton and Kaldor 1987).

    Parameters
    ----------
    num_events : np.ndarray
        Event count in each bin.
    exposure : np.ndarray
        Summed exposure in each bin.
    group_ids : np.ndarray
        Group of each bin, broadcastable to the shape of `num_events`; bins
        in the same group are pooled and share a ratio.
    pseudocount : float, default 1.0
        Prior weight, in expected events, shrinking each ratio toward 1.

    Returns
    -------
    np.ndarray
        Rate ratio of each bin's group, with the shape of `num_events`.
        All ratios are 1.0 if there are no events or no exposure.

    See Also
    --------
    alifestd_estimate_trait_gain_selection_asexual :
        Uses these ratios to rescale measured branch-length exposure by its
        effective exposure, as observed through reference traits.

    References
    ----------
    Clayton, D., and Kaldor, J. (1987). Empirical Bayes estimates of
    age-standardized relative risks for use in disease mapping. Biometrics,
    43(3), 671-681.
    """
    num_events = np.asarray(num_events, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    group_ids = np.broadcast_to(group_ids, num_events.shape).ravel()
    _, group_ids = np.unique(group_ids, return_inverse=True)

    observed = np.bincount(group_ids, weights=num_events.ravel())
    group_exposure = np.bincount(group_ids, weights=exposure.ravel())
    if not observed.sum() or not group_exposure.sum():
        return np.ones_like(num_events)

    expected = group_exposure * (observed.sum() / group_exposure.sum())
    ratio = np.divide(
        observed + pseudocount,
        expected + pseudocount,
        out=np.ones_like(observed),
        where=expected + pseudocount > 0,
    )
    return ratio[group_ids].reshape(num_events.shape)
