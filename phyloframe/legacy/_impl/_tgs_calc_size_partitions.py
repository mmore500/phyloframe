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
    """Implementation detail for `alifestd_tgs_allmdc_estimate_polars`. Acronym
    tgs stands for trait gain/selection.

    Returns each branch's size partition, by clade size on a log scale
    with base `calibrate_size_partition_ratio` (see `calc_geometric_bins`),
    numbered consecutively over occupied bins, so clade sizes absent from
    the tree leave no empty size partitions. Leaves (clade size 1) form
    size partition 0.

    With `calibrate`, inner size partitions go up only while each holds at
    least `calibrate_size_partition_min_neutral_gains` of the neutral trait
    gains on branches at `neutral_gain_positions`; the first with fewer,
    and all above, join the one below, or, if the first inner size
    partition has too few, all inner branches form one size partition.
    Leaves always keep their own size partition.
    """
    clade_sizes = np.asarray(clade_sizes)
    # number occupied bins consecutively, dropping empty ones
    __, size_partitions = np.unique(
        calc_geometric_bins(clade_sizes, ratio=calibrate_size_partition_ratio),
        return_inverse=True,
    )
    num_size_partitions = size_partitions.max() + 1
    neutral_gains = np.bincount(
        size_partitions[neutral_gain_positions],
        minlength=num_size_partitions,
    )
    if calibrate:
        (sparse_inner,) = np.nonzero(
            neutral_gains[1:] < calibrate_size_partition_min_neutral_gains
        )
        if len(sparse_inner):
            # inner size partitions are 1 and up; never join the leaves
            cap = max(sparse_inner[0], 1)
            if cap < num_size_partitions - 1:
                logging.info(
                    "- alifestd_tgs_allmdc_estimate_polars: size partitions "
                    f"above {cap} have fewer than "
                    f"{calibrate_size_partition_min_neutral_gains} neutral "
                    f"gains, so join size partition {cap} for calibration",
                )
            size_partitions = np.minimum(size_partitions, cap)
            num_size_partitions = cap + 1
            neutral_gains = np.bincount(
                size_partitions[neutral_gain_positions],
                minlength=num_size_partitions,
            )

    for size_partition in range(num_size_partitions):
        sizes = clade_sizes[size_partitions == size_partition]
        logging.info(
            "- alifestd_tgs_allmdc_estimate_polars: size partition "
            f"{size_partition} has clade sizes {sizes.min()}-{sizes.max()}, "
            f"{len(sizes)} branches, and {neutral_gains[size_partition]} "
            "neutral trait gains",
        )
    return size_partitions
