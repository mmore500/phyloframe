import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_calc_calibrated_lengths import tgs_calc_calibrated_lengths


def tgs_tabulate_partitions(
    *,
    counts: typing.Sequence[pd.DataFrame],
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
    max_rate_ratio: float,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per trait and partition, trait-major, from per-trait
    `counts` (see `tgs_count_traits`), with columns `trait` (index,
    categorical over all traits), `num_trials`, `num_successes`, and the
    exposure episodes' mean `length` (zero without episodes), calibrated
    against neutral traits if `calibration_neighborhoods` is given, with
    rate ratios capped at `max_rate_ratio` (see
    `tgs_calc_calibrated_lengths`).
    """
    num_traits = len(counts)
    partition_df = pd.concat(
        [
            trait_counts[["num_trials", "num_successes"]].assign(
                length=tgs_calc_calibrated_lengths(
                    counts=counts,
                    trait=trait,
                    neutral_traits=neutral_traits,
                    calibration_neighborhoods=calibration_neighborhoods,
                    max_rate_ratio=max_rate_ratio,
                )[0],
            )
            for trait, trait_counts in enumerate(counts)
        ],
        keys=range(num_traits),
        names=["trait", None],
    ).reset_index(level=0)
    partition_df["trait"] = pd.Categorical(
        partition_df["trait"], categories=range(num_traits)
    )
    return partition_df.reset_index(drop=True)
