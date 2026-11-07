import typing

import numpy as np
import opytional as opyt


def sample_bootstrap_counts(
    num_items: int, *, seed: typing.Optional[int] = None
) -> np.ndarray:
    """Sample how many times each item is drawn in a given bootstrap
    replicate.

    The replicate draws `num_items` items uniformly with replacement, so an
    item may be drawn zero, one, or several times. Weighting each item by
    its count is equivalent to resampling items, without copying data.

    Parameters
    ----------
    num_items : int
        Number of items.
    seed : int, optional
        Seed for the draw; drawn from numpy's global random state if not
        given.

    Returns
    -------
    np.ndarray
        Integer array of length `num_items`, summing to `num_items`.

    Notes
    -----
    Given `seed`, draws only from a generator seeded by it, not global
    random state. So, concurrent seeded calls from different threads are
    reproducible.

    Representing a replicate as multinomial counts over the original items
    is the weighted bootstrap view of Praestgaard and Wellner (1993).

    References
    ----------
    Praestgaard, J., and Wellner, J. A. (1993). Exchangeably weighted
    bootstraps of the general empirical process. Annals of Probability,
    21(4), 2053-2086.
    https://doi.org/10.1214/aop/1176989011
    """
    seed = opyt.or_else(seed, lambda: np.random.randint(2**31))
    if num_items == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n=num_items, pvals=np.full(shape=num_items, fill_value=1 / num_items)
    )
