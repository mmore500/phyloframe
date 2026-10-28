import typing

import numpy as np


def iter_block_bootstrap_weights(
    block_ids: np.ndarray,
    num_replicates: int,
    seed: typing.Optional[int] = None,
) -> typing.Iterator[np.ndarray]:
    """Yield per-item weights for block bootstrap replicates.

    Each replicate resamples blocks with replacement, as many as there are
    blocks; each item's weight is the number of times its block was drawn.
    Weighting avoids materializing resampled data, so memory stays
    proportional to the number of items.

    Parameters
    ----------
    block_ids : np.ndarray
        Block of each item, as integers in `range(num_blocks)`.
    num_replicates : int
        Number of replicates to yield.
    seed : int, optional
        Seed for resampling. Each replicate draws from an independent stream
        spawned from `seed`, so replicates are reproducible regardless of
        how they are consumed (e.g., in parallel).

    Yields
    ------
    np.ndarray
        Float weight of each item, with the shape of `block_ids`.
    """
    block_ids = np.asarray(block_ids)
    num_blocks = block_ids.max() + 1 if block_ids.size else 0
    for seed_seq in np.random.SeedSequence(seed).spawn(num_replicates):
        draws = np.random.default_rng(seed_seq).integers(
            0, num_blocks, num_blocks
        )
        block_weights = np.bincount(draws, minlength=num_blocks)
        yield block_weights[block_ids].astype(np.float64)
