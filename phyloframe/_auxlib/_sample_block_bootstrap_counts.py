import numpy as np

from ._jit import jit


@jit(nopython=True)
def sample_block_bootstrap_counts(
    num_blocks: int, num_replicates: int
) -> np.ndarray:
    """Sample how many times each block is drawn in block bootstrap
    replicates.

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
    """
    if num_blocks == 0:
        return np.zeros((num_replicates, 0), dtype=np.int64)
    return np.random.multinomial(
        num_blocks, np.full(num_blocks, 1 / num_blocks), size=num_replicates
    )
