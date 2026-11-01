import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._TgsAllmdcPartitioning import TgsAllmdcPartitioning
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)
from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_trait_branches import tgs_find_trait_branches

# per partition with branches: branch counts, gain counts, mean (calibrated)
# lengths, and the (sparse) share of branches in each survival-time bin
TgsAllmdcData = typing.Tuple[
    np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix
]


def tgs_calc_allmdc_data(
    *,
    branches: pd.DataFrame,
    partitioning: TgsAllmdcPartitioning,
    unit_weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
) -> typing.List[TgsAllmdcData]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait, for `branches` (see `tgs_calc_branches`), its
    partitions' (weighted) counts of branches that
    started without the trait and of gains, their mean lengths (calibrated
    against neutral traits if `calibrate`), and how those branches spread
    over survival-time bins, omitting partitions without branches.
    """
    trait_units, trait_gains = tgs_find_trait_branches(branches)
    counts = tgs_count_traits(
        leaf_partitions=partitioning.partitions,
        num_partitions=partitioning.num_partitions,
        pendant_lengths=branches["branch_length"].to_numpy(),
        leaf_weights=unit_weights,
        trait_leaves=trait_units,
        trait_carriers=trait_gains,
    )
    shape = (partitioning.num_partitions, len(partitioning.survival_bin_times))
    data = []
    for trait, units in enumerate(trait_units):
        lengths, __ = tgs_calc_calibrated_pendant_lengths(
            counts=counts,
            trait=trait,
            neutral_traits=neutral_traits,
            calibration_neighborhoods=(
                partitioning.calibration_neighborhoods if calibrate else None
            ),
        )
        n = counts.num_leaves[trait]
        keep = n > 0
        # sparse, as partitions span few of many survival-time bins
        survival_shares = sparse.csr_matrix(
            (
                unit_weights[units],
                (
                    partitioning.partitions[units],
                    partitioning.survival_bins[units],
                ),
            ),
            shape=shape,
        )[keep]
        survival_shares = sparse.diags(1.0 / n[keep]) @ survival_shares
        data.append(
            (
                n[keep],
                counts.num_carriers[trait][keep],
                lengths[keep],
                survival_shares,
            ),
        )
    return data
