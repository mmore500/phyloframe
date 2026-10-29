import numpy as np


def sample_bootstrap_counts(num_items: int, seed: int) -> np.ndarray:
    """Sample how many times each item is drawn in a given bootstrap
    replicate.

    The replicate draws `num_items` items uniformly with replacement, so an
    item may be drawn zero, one, or several times. Weighting each item by
    its count is equivalent to resampling items, without copying data.

    Parameters
    ----------
    num_items : int
        Number of items.
    seed : int
        Seed for this replicate's draws. Each call uses its own generator,
        so replicates are reproducible regardless of which thread or process
        draws them, or in what order.

    Returns
    -------
    np.ndarray
        Integer array of length `num_items`, summing to `num_items`.

    Notes
    -----
    Representing a replicate as multinomial counts over the original items
    is the weighted bootstrap view of Praestgaard and Wellner (1993).

    References
    ----------
    Praestgaard, J., and Wellner, J. A. (1993). Exchangeably weighted
    bootstraps of the general empirical process. Annals of Probability,
    21(4), 2053-2086.
    https://doi.org/10.1214/aop/1176989011
    """
    if num_items == 0:
        return np.zeros(0, dtype=np.int64)
    return np.random.default_rng(seed).multinomial(
        num_items, np.full(num_items, 1 / num_items)
    )
