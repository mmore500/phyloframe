import typing

import numpy as np

from ._assign_kdtree_partitions import assign_kdtree_partitions
from ._calc_tiebroken_ranks import calc_tiebroken_ranks
from ._ceil_div import ceil_div


def assign_rank_bins(
    coordinates: typing.Sequence[np.ndarray],
    *,
    max_bins: int,
    min_bin_size: int,
) -> typing.Tuple[np.ndarray, int]:
    """Assign items to equal-occupancy bins by their ranked coordinates.

    Bins are the leaves of a balanced k-d tree over items' tiebroken ranks
    (see `calc_tiebroken_ranks`), so bins do not depend on item order unless
    items tie on every coordinate. Uses the largest power of two of bins, up
    to `max_bins`, that keeps at least `min_bin_size` items in each.

    Parameters
    ----------
    coordinates : Sequence[np.ndarray]
        Arrays of equal length, one per coordinate.
    max_bins : int
        Largest number of bins; a power of two.
    min_bin_size : int
        Fewest items per bin, unless there are fewer items in all.

    Returns
    -------
    bins : np.ndarray
        Bin of each item, as integers numbered from 0.
    num_bins : int
        Number of bins.
    """
    assert int(max_bins).bit_count() == 1
    assert min_bin_size >= 1
    num_items = len(coordinates[0])
    if num_items == 0:
        return np.zeros(0, dtype=np.int64), 0

    ranks = calc_tiebroken_ranks(coordinates) / num_items  # scaled to (0, 1]
    num_levels = int(
        np.clip(np.log2(num_items / min_bin_size), 0, np.log2(max_bins))
    )
    num_bins = 2**num_levels
    # median splitting stops exactly `num_levels` levels down, as nodes there
    # hold at most this many items and nodes above hold more
    bins = assign_kdtree_partitions(ranks, ceil_div(num_items, num_bins))
    return bins, num_bins
