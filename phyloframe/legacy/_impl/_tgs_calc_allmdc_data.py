import typing

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import preprocessing

from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_trait_exposures import tgs_find_trait_exposures
from ._tgs_tabulate_partitions import tgs_tabulate_partitions


def tgs_calc_allmdc_data(
    *,
    exposure_df: pd.DataFrame,
    partitioning: typing.Dict[str, typing.Any],
    weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibrate_max_rate_ratio: float,
) -> typing.Tuple[pd.DataFrame, sparse.csr_matrix]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Summarizes `exposure_df` (see `tgs_calc_exposure_df`) by its
    `partitioning` (see `tgs_partition_allmdc_exposures`). Returns
    `partition_df`, with one row per trait and partition with branches
    starting without the trait, with columns `trait` (index, categorical
    over all traits), the (weighted) counts of those branches,
    `num_trials`, and of gains, `num_successes`, and their mean `length`
    (calibrated against neutral traits if `calibrate`). Also returns
    `survival_shares`, a sparse matrix with a row per `partition_df` row and
    a column per distinct survival time, giving the share of the row's
    branches
    in each bin.
    """
    partitions = partitioning["partitions"]
    num_partitions = partitions.max() + 1
    trait_units, trait_gains = tgs_find_trait_exposures(exposure_df)
    num_traits = len(trait_units)
    counts = tgs_count_traits(
        leaf_partitions=partitions,
        num_partitions=num_partitions,
        pendant_lengths=exposure_df["branch_length"].to_numpy(),
        leaf_weights=weights,
        trait_leaves=trait_units,
        trait_carriers=trait_gains,
    )
    # one row per trait and partition, before dropping rows without branches
    all_partition_df = tgs_tabulate_partitions(
        counts=counts,
        neutral_traits=neutral_traits,
        calibration_neighborhoods=(
            partitioning["calibration_neighborhoods"] if calibrate else None
        ),
        max_rate_ratio=calibrate_max_rate_ratio,
    )
    # sparse, as partitions span few of many distinct survival times; each
    # trait's branches go in that trait's block of rows
    units = np.concatenate(trait_units)
    unit_traits = np.repeat(np.arange(num_traits), list(map(len, trait_units)))
    all_survival_shares = preprocessing.normalize(
        sparse.csr_matrix(
            (
                weights[units],
                (
                    unit_traits * num_partitions + partitions[units],
                    partitioning["survival_time_indices"][units],
                ),
            ),
            shape=(
                num_traits * num_partitions,
                len(partitioning["distinct_survival_times"]),
            ),
        ),
        norm="l1",
    )
    has_branches = all_partition_df["num_trials"].to_numpy() > 0
    return (
        all_partition_df[has_branches].reset_index(drop=True),
        all_survival_shares[has_branches],
    )
