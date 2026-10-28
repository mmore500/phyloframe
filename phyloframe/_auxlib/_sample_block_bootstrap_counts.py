import numpy as np

from ._jit import jit


@jit(nopython=True)
def sample_block_bootstrap_counts(
    num_blocks: int, num_replicates: int
) -> np.ndarray:
    """Sample how many times each block is drawn in block bootstrap
    replicates.

    Resampling whole blocks of correlated items (e.g., clades, spatial
    tiles, or contiguous segments of a time series), rather than individual
    items, keeps correlation within blocks in each replicate, so bootstrap
    standard errors reflect it instead of understating uncertainty as if
    items were independent.

    Each replicate draws `num_blocks` blocks uniformly with replacement, so
    a block may be drawn zero, one, or several times. Weighting each item by
    its block's count is equivalent to resampling items block-wise, without
    copying data.

    Parameters
    ----------
    num_blocks : int
        Number of blocks.
    num_replicates : int
        Number of bootstrap replicates.

    Returns
    -------
    np.ndarray
        Integer array of shape (num_replicates, num_blocks), each row summing
        to `num_blocks`.

    Notes
    -----
    Draws from numpy's global random state within jit-compiled code, so
    `RngStateContext` and `with_rng_state_context` seed it.

    Resampling labeled groups of correlated items is also called the
    cluster bootstrap (Field and Welsh 2007; Cameron et al. 2008); block
    bootstrap usually names resampling contiguous blocks of a time series
    (Kunsch 1989) or of space (Lahiri 2003). Representing replicates as
    multinomial counts over the original items is the weighted bootstrap
    view of Praestgaard and Wellner (1993).

    References
    ----------
    Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). Bootstrap-based
    improvements for inference with clustered errors. Review of Economics
    and Statistics, 90(3), 414-427.
    https://doi.org/10.1162/rest.90.3.414

    Field, C. A., and Welsh, A. H. (2007). Bootstrapping clustered data.
    Journal of the Royal Statistical Society Series B, 69(3), 369-390.
    https://doi.org/10.1111/j.1467-9868.2007.00593.x

    Kunsch, H. R. (1989). The jackknife and the bootstrap for general
    stationary observations. Annals of Statistics, 17(3), 1217-1241.
    https://doi.org/10.1214/aos/1176347265

    Lahiri, S. N. (2003). Resampling Methods for Dependent Data. Springer.
    https://doi.org/10.1007/978-1-4757-3803-2

    Praestgaard, J., and Wellner, J. A. (1993). Exchangeably weighted
    bootstraps of the general empirical process. Annals of Probability,
    21(4), 2053-2086.
    https://doi.org/10.1214/aop/1176989011
    """
    if num_blocks == 0:
        return np.zeros((num_replicates, 0), dtype=np.int64)
    return np.random.multinomial(
        num_blocks, np.full(num_blocks, 1 / num_blocks), size=num_replicates
    )
