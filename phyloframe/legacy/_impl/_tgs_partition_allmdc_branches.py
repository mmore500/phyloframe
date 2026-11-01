import logging
import typing

import numpy as np
from scipy import sparse

from ._TgsAllmdcBranches import TgsAllmdcBranches
from ._TgsAllmdcPartitioning import TgsAllmdcPartitioning
from ._tgs_count_traits import tgs_count_traits
from ._tgs_partition_branches import tgs_partition_branches
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def tgs_partition_allmdc_branches(
    *,
    branches: TgsAllmdcBranches,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    clade_size_calibration_ratio: float,
    min_calibration_block_neutral_gains: int,
    num_survival_time_bins: int,
) -> TgsAllmdcPartitioning:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Splits branches into blocks by clade size on a log scale, with base
    `clade_size_calibration_ratio`: block ceil(log_ratio(clade size)), so
    a ratio of 2 gives blocks of 1, 2, 3-4, 5-8, 9-16, ... leaves. With
    `calibrate`, blocks go up only while each has at least
    `min_calibration_block_neutral_gains` neutral trait gains; the first
    block with fewer, and all larger clades, join the block below, so every
    block has neutral gains to calibrate against.

    Within each block, partitions branches into up to
    `num_calibration_partitions` equal-occupancy partitions by ranked length
    and origin time (skipping either if constant) and, with `calibrate`,
    selects calibration neighborhoods, which so never mix blocks, as
    neutral gains per unit length may differ with clade size. Survival time
    is left out, as the model accounts for it. Also bins survival times
    into up to `num_survival_time_bins` equal-occupancy bins.
    """
    # tolerance keeps exact powers of the ratio, e.g., 4 for ratio 2, in
    # the block they close despite floating point error
    blocks = np.ceil(
        np.log(branches.clade_sizes) / np.log(clade_size_calibration_ratio)
        - 1e-9
    ).astype(np.int64)
    if calibrate:
        neutral_gains = np.bincount(
            blocks[
                np.concatenate(
                    [branches.trait_gains[r] for r in neutral_traits]
                )
            ],
            minlength=blocks.max() + 1,
        )
        (sparse_blocks,) = np.nonzero(
            neutral_gains < min_calibration_block_neutral_gains
        )
        # leaves always form their own block
        cap = max(sparse_blocks[0] - 1, 0) if len(sparse_blocks) else None
        if cap is not None and cap < blocks.max():
            logging.info(
                "- alifestd_tgs_allmdc_estimate_asexual: clade-size blocks "
                f"above {cap} have fewer than "
                f"{min_calibration_block_neutral_gains} neutral gains, so "
                f"join block {cap} for calibration",
            )
            blocks = np.minimum(blocks, cap)
    partitions = np.empty(len(blocks), dtype=np.int64)
    neighborhoods = []
    num_partitions = 0
    for block in np.unique(blocks):
        (members,) = np.nonzero(blocks == block)
        # not survival time, which the model accounts for: calibrating along
        # it would absorb its effect on neutral gains (e.g., from loss) into
        # branch length, then count it again in the survival term
        coordinates = [
            branches.branch_lengths[members],
            branches.origin_times[members],
        ]
        (
            block_partitions,
            num_block_partitions,
            coords,
        ) = tgs_partition_branches(
            # constant coordinates would only split on ties
            coordinates=[x for x in coordinates if np.ptp(x) > 0]
            or coordinates,
            max_partitions=num_calibration_partitions,
        )
        partitions[members] = block_partitions + num_partitions
        num_partitions += num_block_partitions
        if calibrate:
            # positions, within the block, of each trait's members
            def subset(positions: np.ndarray) -> np.ndarray:
                return np.searchsorted(
                    members, positions[np.isin(positions, members)]
                )

            neighborhoods.append(
                tgs_select_calibration_neighborhoods(
                    partition_coords=coords,
                    neighborhood_fractions=calibration_neighborhood_fractions,
                    counts=tgs_count_traits(
                        leaf_partitions=block_partitions,
                        num_partitions=num_block_partitions,
                        pendant_lengths=branches.branch_lengths[members],
                        leaf_weights=np.ones(len(members)),
                        trait_leaves=list(map(subset, branches.trait_units)),
                        trait_carriers=list(map(subset, branches.trait_gains)),
                    ),
                    neutral_traits=neutral_traits,
                    trait_names=trait_names,
                ),
            )

    edges = np.unique(
        np.quantile(
            branches.survival_times,
            np.linspace(0, 1, num_survival_time_bins + 1),
        ),
    )
    num_bins = max(len(edges) - 1, 1)
    survival_bins = np.clip(
        np.searchsorted(edges, branches.survival_times, side="right") - 1,
        0,
        num_bins - 1,
    )
    survival_bin_times = np.bincount(
        survival_bins, branches.survival_times, minlength=num_bins
    ) / np.maximum(np.bincount(survival_bins, minlength=num_bins), 1)
    return TgsAllmdcPartitioning(
        partitions=partitions,
        num_partitions=num_partitions,
        survival_bins=survival_bins,
        survival_bin_times=survival_bin_times,
        calibration_neighborhoods=(
            sparse.block_diag(neighborhoods, format="csr")
            if calibrate
            else None
        ),
    )
