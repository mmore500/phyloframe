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
    weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
) -> typing.Tuple[pd.DataFrame, sparse.csr_matrix]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Summarizes `branches` (see `tgs_calc_branches`) by their `partitioning`
    (see `tgs_partition_allmdc_branches`). Returns a DataFrame with one row
    per trait and partition with branches starting without the trait,
    with columns `trait` (index, categorical over all traits), the
    (weighted) counts of those branches, `num_trials`, and of gains,
    `num_successes`, and their mean `length` (calibrated against neutral
    traits if `calibrate`). Also returns `survival_shares`, a sparse matrix
    with a row per DataFrame row and a column per survival-time bin, giving
    the share of the partition's branches in each bin.
    """
    partitions = partitioning["partitions"]
    num_partitions = partitions.max() + 1
    trait_units, trait_gains = tgs_find_trait_branches(branches)
    counts = tgs_count_traits(
        leaf_partitions=partitions,
        num_partitions=num_partitions,
        pendant_lengths=branches["branch_length"].to_numpy(),
        leaf_weights=weights,
        trait_leaves=trait_units,
        trait_carriers=trait_gains,
    )
    shape = (num_partitions, len(partitioning["survival_bin_times"]))
    frames, shares = [], []
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
        frames.append(
            pd.DataFrame(
                {
                    "trait": trait,
                    "num_trials": n[keep],
                    "num_successes": counts.num_carriers[trait][keep],
                    "length": lengths[keep],
                },
            ),
        )
        # sparse, as partitions span few of many survival-time bins
        trait_shares = sparse.csr_matrix(
            (
                weights[units],
                (partitions[units], partitioning["survival_bins"][units]),
            ),
            shape=shape,
        )[keep]
        shares.append(sparse.diags(1.0 / n[keep]) @ trait_shares)
    data = pd.concat(frames, ignore_index=True)
    data["trait"] = pd.Categorical(
        data["trait"], categories=range(len(trait_units))
    )
    return data, sparse.vstack(shares, format="csr")
