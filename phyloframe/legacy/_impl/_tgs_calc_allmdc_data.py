import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)
from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_trait_branches import tgs_find_trait_branches


def tgs_calc_allmdc_data(
    *,
    branches: pd.DataFrame,
    partitioning: typing.Dict[str, typing.Any],
    unit_weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait, for `branches` (see `tgs_calc_branches`) and their
    `partitioning` (see `tgs_partition_allmdc_branches`), a dict of arrays
    over partitions with branches starting without the trait, with
    (weighted) counts of those branches, `num_trials`, and of gains,
    `num_successes`; their mean `lengths` (calibrated against neutral
    traits if `calibrate`); and `survival_shares`, a sparse
    partition-by-survival-time-bin matrix of the share of each partition's
    branches in each bin.
    """
    partitions = partitioning["partitions"]
    num_partitions = partitions.max() + 1
    trait_units, trait_gains = tgs_find_trait_branches(branches)
    counts = tgs_count_traits(
        leaf_partitions=partitions,
        num_partitions=num_partitions,
        pendant_lengths=branches["branch_length"].to_numpy(),
        leaf_weights=unit_weights,
        trait_leaves=trait_units,
        trait_carriers=trait_gains,
    )
    shape = (num_partitions, len(partitioning["survival_bin_times"]))
    data = []
    for trait, units in enumerate(trait_units):
        lengths, __ = tgs_calc_calibrated_pendant_lengths(
            counts=counts,
            trait=trait,
            neutral_traits=neutral_traits,
            calibration_neighborhoods=(
                partitioning["calibration_neighborhoods"]
                if calibrate
                else None
            ),
        )
        n = counts.num_leaves[trait]
        keep = n > 0
        # sparse, as partitions span few of many survival-time bins
        survival_shares = sparse.csr_matrix(
            (
                unit_weights[units],
                (
                    partitions[units],
                    partitioning["survival_bins"][units],
                ),
            ),
            shape=shape,
        )[keep]
        survival_shares = sparse.diags(1.0 / n[keep]) @ survival_shares
        data.append(
            {
                "num_trials": n[keep],
                "num_successes": counts.num_carriers[trait][keep],
                "lengths": lengths[keep],
                "survival_shares": survival_shares,
            },
        )
    return data
