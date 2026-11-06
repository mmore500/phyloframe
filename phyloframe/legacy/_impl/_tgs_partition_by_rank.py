import logging
import typing

import numpy as np

from ..._auxlib._assign_kdtree_partitions import assign_kdtree_partitions
from ..._auxlib._calc_group_means import calc_group_means
from ..._auxlib._calc_tiebroken_ranks import calc_tiebroken_ranks
from ..._auxlib._ceil_div import ceil_div


def tgs_partition_by_rank(
    *,
    coordinates: typing.Sequence[np.ndarray],
    max_partitions: int,
) -> typing.Tuple[np.ndarray, int, np.ndarray]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns partition of each branch, number of partitions, and each
    partition's mean coordinates, grouping branches into equal-occupancy
    partitions by their ranked `coordinates` (e.g., branch length and
    origin time). Uses up to `max_partitions` partitions, a power of two,
    but keeps at least 4 branches in each.
    """
    num_branches = len(coordinates[0])
    # tie-breaking (e.g., leaves' survival times, all 0) by the other
    # coordinates keeps partitions independent of row order
    # scaled to (0, 1]
    ranks = calc_tiebroken_ranks(coordinates) / num_branches
    num_levels = int(
        np.clip(np.log2(num_branches / 4), 0, np.log2(max_partitions))
    )
    num_partitions = 2**num_levels
    # median splitting stops exactly `num_levels` levels down, as nodes there
    # hold at most this many branches and nodes above, with at least 4 per
    # partition, hold more
    partition_size = ceil_div(num_branches, num_partitions)
    partitions = assign_kdtree_partitions(ranks, partition_size)
    partition_sizes = np.bincount(partitions, minlength=num_partitions)
    logging.info(
        "- tgs_partition_by_rank: "
        f"partitioned {num_branches} branches into {num_partitions} "
        f"partitions ({num_levels} levels of splits), of sizes "
        f"{np.unique(partition_sizes).tolist()}",
    )
    partition_coords = calc_group_means(
        partitions, ranks, num_groups=num_partitions
    )
    return partitions, num_partitions, partition_coords
