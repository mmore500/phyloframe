import typing

import numpy as np

from ._tgs_partition_branches import tgs_partition_branches


def tgs_partition_leaves(
    *,
    pendant_lengths: np.ndarray,
    origin_times: np.ndarray,
    max_partitions: int,
) -> typing.Tuple[np.ndarray, int, np.ndarray]:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns partition of each leaf, number of partitions, and each
    partition's mean coordinates, grouping leaves into equal-occupancy
    partitions by ranked pendant length and origin time. Uses up to
    `max_partitions` partitions, a power of two, but keeps at least 4
    leaves in each. See `tgs_partition_branches`.
    """
    return tgs_partition_branches(
        coordinates=[pendant_lengths, origin_times],
        max_partitions=max_partitions,
    )
