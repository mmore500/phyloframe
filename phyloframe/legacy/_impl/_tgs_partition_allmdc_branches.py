import logging
import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_trait_branches import tgs_find_trait_branches
from ._tgs_partition_branches import tgs_partition_branches
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def tgs_partition_allmdc_branches(
    *,
    branches: pd.DataFrame,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    calibrate: bool,
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
    calibrate_size_partition_ratio: float,
    calibrate_size_partition_min_neutral_gains: int,
    num_survival_time_bins: int,
) -> typing.Dict[str, typing.Any]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Splits `branches` (see `tgs_calc_branches`) into size partitions by
    clade size on a log scale, with base `calibrate_size_partition_ratio`:
    size partition ceil(log_ratio(clade size)), so a ratio of 2 gives size
    partitions of 1, 2, 3-4, 5-8, 9-16, ... leaves. With `calibrate`, size
    partitions go up only while each has at least
    `calibrate_size_partition_min_neutral_gains` neutral trait gains; the
    first with fewer, and all larger clades, join the one below, so every
    size partition has neutral gains to calibrate against.

    Within each size partition, partitions branches into up to
    `calibrate_num_partitions` equal-occupancy partitions by ranked length
    and origin time (skipping either if constant) and, with `calibrate`,
    selects calibration neighborhoods, which so never mix size partitions,
    as neutral gains per unit length may differ with clade size. Also bins
    survival times into up to `num_survival_time_bins` equal-occupancy
    bins.

    Returns a dict with each branch's `partitions` index (numbered
    consecutively across size partitions) and `survival_bins` index, each
    bin's mean survival time `survival_bin_times`, and the
    partition-by-partition `calibration_neighborhoods` (None without
    `calibrate`).
    """
    branch_lengths = branches["branch_length"].to_numpy()
    origin_times = branches["origin_time"].to_numpy()
    survival_times = branches["survival_time"].to_numpy()
    trait_units, trait_gains = tgs_find_trait_branches(branches)
    clade_sizes = branches["clade_size"].to_numpy()
    # size partition k holds clade sizes in (ratio ** (k - 1), ratio ** k]
    size_partition_bounds = calibrate_size_partition_ratio ** np.arange(
        np.log(clade_sizes.max()) // np.log(calibrate_size_partition_ratio) + 2
    )
    size_partitions = np.searchsorted(size_partition_bounds, clade_sizes)
    neutral_gain_positions = np.concatenate(
        [trait_gains[r] for r in neutral_traits] or [np.empty(0, dtype=int)]
    )
    if calibrate:
        neutral_gains = np.bincount(
            size_partitions[neutral_gain_positions],
            minlength=size_partitions.max() + 1,
        )
        (sparse_size_partitions,) = np.nonzero(
            neutral_gains < calibrate_size_partition_min_neutral_gains
        )
        # leaves always form their own size partition
        cap = (
            max(sparse_size_partitions[0] - 1, 0)
            if len(sparse_size_partitions)
            else None
        )
        if cap is not None and cap < size_partitions.max():
            logging.info(
                "- alifestd_tgs_allmdc_estimate_asexual: size partitions "
                f"above {cap} have fewer than "
                f"{calibrate_size_partition_min_neutral_gains} neutral gains, "
                f"so join size partition {cap} for calibration",
            )
            size_partitions = np.minimum(size_partitions, cap)
    partitions = np.empty(len(size_partitions), dtype=np.int64)
    neighborhoods = []
    num_partitions = 0
    for size_partition in np.unique(size_partitions):
        (members,) = np.nonzero(size_partitions == size_partition)
        # not survival time, which the model accounts for: calibrating along
        # it would absorb its effect on neutral gains (e.g., from loss) into
        # branch length, then count it again in the survival term
        coordinates = [branch_lengths[members], origin_times[members]]
        (
            member_partitions,
            num_member_partitions,
            coords,
        ) = tgs_partition_branches(
            # constant coordinates would only split on ties
            coordinates=[x for x in coordinates if np.ptp(x) > 0]
            or coordinates,
            max_partitions=calibrate_num_partitions,
        )
        partitions[members] = member_partitions + num_partitions
        num_partitions += num_member_partitions
        logging.info(
            "- alifestd_tgs_allmdc_estimate_asexual: size partition "
            f"{size_partition} (clade sizes {clade_sizes[members].min()}"
            f"-{clade_sizes[members].max()}) has {len(members)} branches, "
            f"{np.isin(neutral_gain_positions, members).sum()} neutral trait "
            f"gains, and {num_member_partitions} partitions",
        )
        if calibrate:
            # positions, within the size partition, of each trait's members
            def subset(positions: np.ndarray) -> np.ndarray:
                return np.searchsorted(
                    members, positions[np.isin(positions, members)]
                )

            neighborhoods.append(
                tgs_select_calibration_neighborhoods(
                    partition_coords=coords,
                    neighborhood_fractions=calibrate_neighborhood_fractions,
                    counts=tgs_count_traits(
                        leaf_partitions=member_partitions,
                        num_partitions=num_member_partitions,
                        pendant_lengths=branch_lengths[members],
                        leaf_weights=np.ones(len(members)),
                        trait_leaves=list(map(subset, trait_units)),
                        trait_carriers=list(map(subset, trait_gains)),
                    ),
                    neutral_traits=neutral_traits,
                    trait_names=trait_names,
                ),
            )

    edges = np.unique(
        np.quantile(
            survival_times, np.linspace(0, 1, num_survival_time_bins + 1)
        ),
    )
    num_bins = max(len(edges) - 1, 1)
    survival_bins = np.clip(
        np.searchsorted(edges, survival_times, side="right") - 1,
        0,
        num_bins - 1,
    )
    bin_occupancy = np.bincount(survival_bins, minlength=num_bins)
    survival_bin_times = np.bincount(
        survival_bins, survival_times, minlength=num_bins
    ) / np.maximum(bin_occupancy, 1)
    logging.info(
        "- alifestd_tgs_allmdc_estimate_asexual: binned survival times into "
        f"{num_bins} bins, holding {bin_occupancy.min()}"
        f"-{bin_occupancy.max()} branches each",
    )
    return {
        "partitions": partitions,
        "survival_bins": survival_bins,
        "survival_bin_times": survival_bin_times,
        "calibration_neighborhoods": (
            sparse.block_diag(neighborhoods, format="csr")
            if calibrate
            else None
        ),
    }
