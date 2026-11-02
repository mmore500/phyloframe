import logging

import numpy as np

from ..._auxlib._calc_geometric_bins import calc_geometric_bins


def tgs_calc_size_partitions(
    *,
    clade_sizes: np.ndarray,
    neutral_gain_positions: np.ndarray,
    calibrate: bool,
    calibrate_size_partition_ratio: float,
    calibrate_size_partition_min_neutral_gains: int,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns each branch's size partition, by clade size on a log scale
    with base `calibrate_size_partition_ratio` (see `calc_geometric_bins`).
    With `calibrate`, size partitions go up only while each holds at least
    `calibrate_size_partition_min_neutral_gains` of the neutral trait gains
    on branches at `neutral_gain_positions`; the first with fewer, and all
    above, join the one below, so every size partition has neutral gains
    to calibrate against. Leaves always keep their own size partition.
    """
    size_partitions = calc_geometric_bins(
        clade_sizes, ratio=calibrate_size_partition_ratio
    )
    if not calibrate:
        return size_partitions

    neutral_gains = np.bincount(
        size_partitions[neutral_gain_positions],
        minlength=size_partitions.max() + 1,
    )
    (sparse_size_partitions,) = np.nonzero(
        neutral_gains < calibrate_size_partition_min_neutral_gains
    )
    if not len(sparse_size_partitions):
        return size_partitions

    cap = max(sparse_size_partitions[0] - 1, 0)
    if cap < size_partitions.max():
        logging.info(
            "- alifestd_tgs_allmdc_estimate_asexual: size partitions above "
            f"{cap} have fewer than "
            f"{calibrate_size_partition_min_neutral_gains} neutral gains, so "
            f"join size partition {cap} for calibration",
        )
    return np.minimum(size_partitions, cap)
