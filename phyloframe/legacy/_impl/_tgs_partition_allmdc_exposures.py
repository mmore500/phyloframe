import typing

import numpy as np
import pandas as pd

from ._tgs_bin_survival_times import tgs_bin_survival_times
from ._tgs_calc_size_partitions import tgs_calc_size_partitions
from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_trait_exposures import tgs_find_trait_exposures
from ._tgs_partition_size_partitions import tgs_partition_size_partitions
from ._tgs_select_allmdc_calibration_neighborhoods import (
    tgs_select_allmdc_calibration_neighborhoods,
)


def tgs_partition_allmdc_exposures(
    *,
    exposure_df: pd.DataFrame,
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

    Splits `exposure_df` (see `tgs_calc_exposure_df`) into size partitions by
    clade size (see `tgs_calc_size_partitions`), then each into
    equal-occupancy partitions (see `tgs_partition_size_partitions`), and,
    with `calibrate`, selects calibration neighborhoods within each size
    partition (see `tgs_select_allmdc_calibration_neighborhoods`). Also bins
    survival times (see `tgs_bin_survival_times`).

    Returns a dict with each branch's `partitions` index (numbered
    consecutively across size partitions) and `survival_bins` index, each
    bin's mean survival time `survival_bin_times`, and the
    partition-by-partition `calibration_neighborhoods` (None without
    `calibrate`).
    """
    branch_lengths = exposure_df["branch_length"].to_numpy()
    trait_units, trait_gains = tgs_find_trait_exposures(exposure_df)
    neutral_gain_positions = np.concatenate(
        [trait_gains[r] for r in neutral_traits] or [np.empty(0, dtype=int)]
    )
    size_partitions = tgs_calc_size_partitions(
        clade_sizes=exposure_df["clade_size"].to_numpy(),
        neutral_gain_positions=neutral_gain_positions,
        calibrate=calibrate,
        calibrate_size_partition_ratio=calibrate_size_partition_ratio,
        calibrate_size_partition_min_neutral_gains=(
            calibrate_size_partition_min_neutral_gains
        ),
    )
    partitions, blocks = tgs_partition_size_partitions(
        size_partitions=size_partitions,
        branch_lengths=branch_lengths,
        origin_times=exposure_df["origin_time"].to_numpy(),
        max_partitions=calibrate_num_partitions,
    )
    survival_bins, survival_bin_times = tgs_bin_survival_times(
        exposure_df["survival_time"].to_numpy(),
        max_bins=num_survival_time_bins,
    )
    return {
        "partitions": partitions,
        "survival_bins": survival_bins,
        "survival_bin_times": survival_bin_times,
        "calibration_neighborhoods": (
            tgs_select_allmdc_calibration_neighborhoods(
                counts=tgs_count_traits(
                    leaf_partitions=partitions,
                    num_partitions=partitions.max() + 1,
                    pendant_lengths=branch_lengths,
                    leaf_weights=np.ones(len(partitions)),
                    trait_leaves=trait_units,
                    trait_carriers=trait_gains,
                ),
                blocks=blocks,
                neighborhood_fractions=calibrate_neighborhood_fractions,
                neutral_traits=neutral_traits,
                trait_names=trait_names,
            )
            if calibrate
            else None
        ),
    }
