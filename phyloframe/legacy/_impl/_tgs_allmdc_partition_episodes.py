import typing

import numpy as np
import pandas as pd

from ._tgs_allmdc_partition_by_rank_within_sizes import (
    tgs_allmdc_partition_by_rank_within_sizes,
)
from ._tgs_allmdc_select_calibration_neighborhoods import (
    tgs_allmdc_select_calibration_neighborhoods,
)
from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_trait_exposures import tgs_find_trait_exposures


def tgs_allmdc_partition_episodes(
    *,
    exposure_df: pd.DataFrame,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
    calibrate_max_rate_ratio: float,
) -> typing.Dict[str, typing.Any]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_polars`. Acronym
    tgs stands for trait gain/selection.

    Splits `exposure_df`'s exposure episodes (see `tgs_calc_exposure_df`)
    by their `size_partition` column, if any (else, all are in one size
    partition), then each size partition into equal-occupancy partitions
    (see `tgs_allmdc_partition_by_rank_within_sizes`), and selects
    calibration neighborhoods within each size partition (see
    `tgs_allmdc_select_calibration_neighborhoods`). Also bins episodes by
    survival time, exactly, as trees typically have few distinct survival
    times (e.g., from integer generations).

    Returns a dict with each episode's `partitions` index (numbered
    consecutively across size partitions) and `survival_time_indices`
    index into the sorted `distinct_survival_times`, and the
    partition-by-partition `calibration_neighborhoods`.
    """
    branch_lengths = exposure_df["branch_length"].to_numpy()
    trait_units, trait_gains = tgs_find_trait_exposures(exposure_df)
    # number size partitions consecutively; without them, all are one
    __, size_partitions = np.unique(
        exposure_df.get(
            "size_partition", np.zeros(len(exposure_df), dtype=np.int64)
        ),
        return_inverse=True,
    )
    partitions, blocks = tgs_allmdc_partition_by_rank_within_sizes(
        size_partitions=size_partitions,
        branch_lengths=branch_lengths,
        origin_times=exposure_df["origin_time"].to_numpy(),
        max_partitions=calibrate_num_partitions,
    )
    distinct_survival_times, survival_time_indices = np.unique(
        exposure_df["survival_time"].to_numpy(), return_inverse=True
    )
    return {
        "partitions": partitions,
        "survival_time_indices": survival_time_indices,
        "distinct_survival_times": distinct_survival_times,
        "calibration_neighborhoods": tgs_allmdc_select_calibration_neighborhoods(
                counts=tgs_count_traits(
                    partitions=partitions,
                    num_partitions=partitions.max() + 1,
                    lengths=branch_lengths,
                    weights=np.ones(len(partitions)),
                    trait_episodes=trait_units,
                    trait_gains=trait_gains,
                ),
                blocks=blocks,
                neighborhood_fractions=calibrate_neighborhood_fractions,
                neutral_traits=neutral_traits,
                trait_names=trait_names,
                max_rate_ratio=calibrate_max_rate_ratio,
        ),
    }
