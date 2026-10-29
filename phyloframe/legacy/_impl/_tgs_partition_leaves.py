import logging
import typing

import numpy as np
from scipy import stats

from ..._auxlib._calc_group_means import calc_group_means
from ..._auxlib._calc_kdtree_partitions import calc_kdtree_partitions


def tgs_partition_leaves(
    *,
    pendant_lengths: np.ndarray,
    origin_times: np.ndarray,
    max_partitions: int,
) -> typing.Tuple[np.ndarray, int, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns partition of each leaf, number of partitions, and each
    partition's mean coordinates, grouping leaves into equal-occupancy
    partitions by ranked pendant length and origin time. Uses up to
    `max_partitions` partitions, a power of two, but keeps at least 4
    leaves in each.
    """
    num_leaves = len(pendant_lengths)
    leaf_ranks = np.column_stack(
        [
            stats.rankdata(x, method="ordinal") / num_leaves
            for x in (pendant_lengths, origin_times)
        ],
    )
    num_levels = int(
        max(0, min(np.log2(max_partitions), np.log2(num_leaves / 4)))
    )
    num_partitions = 2**num_levels
    # median splitting stops exactly `num_levels` levels down, as nodes there
    # hold at most this many leaves and nodes above, with at least 4 per
    # partition, hold more
    partition_size = -(-num_leaves // num_partitions)  # ceiling division
    leaf_partitions = calc_kdtree_partitions(leaf_ranks, partition_size)
    partition_sizes = np.bincount(leaf_partitions, minlength=num_partitions)
    logging.info(
        "- alifestd_estimate_trait_gain_selection_pendant_asexual: "
        f"partitioned {num_leaves} leaves into {num_partitions} partitions "
        f"({num_levels} levels of splits), of sizes "
        f"{np.unique(partition_sizes).tolist()}",
    )
    partition_coords = calc_group_means(
        leaf_partitions, leaf_ranks, num_groups=num_partitions
    )
    return leaf_partitions, num_partitions, partition_coords
