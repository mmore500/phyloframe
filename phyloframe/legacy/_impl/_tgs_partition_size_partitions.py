import logging
import typing

import numpy as np

from ._tgs_calc_rank_partitions import tgs_calc_rank_partitions


def tgs_partition_size_partitions(
    *,
    size_partitions: np.ndarray,
    branch_lengths: np.ndarray,
    origin_times: np.ndarray,
    max_partitions: int,
) -> typing.Tuple[np.ndarray, typing.List[typing.Tuple[int, int, np.ndarray]]]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_polars`. Acronym
    tgs stands for trait gain/selection.

    Partitions each size partition's branches into up to `max_partitions`
    equal-occupancy partitions by ranked length and origin time, skipping
    either if constant (see `tgs_calc_rank_partitions`). Returns each
    branch's partition, numbered consecutively across size partitions, and,
    per size partition, its first partition, number of partitions, and
    their mean coordinates.
    """
    partitions = np.empty(len(size_partitions), dtype=np.int64)
    blocks = []
    offset = 0
    for size_partition in np.unique(size_partitions):
        (members,) = np.nonzero(size_partitions == size_partition)
        # not survival time, which the model accounts for: calibrating along
        # it would absorb its effect on neutral gains (e.g., from loss) into
        # branch length, then count it again in the survival term
        coordinates = [branch_lengths[members], origin_times[members]]
        member_partitions, num, coords = tgs_calc_rank_partitions(
            # constant coordinates would only split on ties
            coordinates=[x for x in coordinates if np.ptp(x) > 0]
            or coordinates,
            max_partitions=max_partitions,
        )
        partitions[members] = member_partitions + offset
        blocks.append((offset, num, coords))
        offset += num
        logging.info(
            "- alifestd_tgs_allmdc_estimate_polars: size partition "
            f"{size_partition} has {len(members)} branches in {num} "
            "partitions",
        )
    return partitions, blocks
