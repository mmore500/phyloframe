import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._TgsCounts import TgsCounts
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)


def tgs_tabulate_partitions(
    *,
    counts: TgsCounts,
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait and partition, trait-major, with columns
    `trait` (index, categorical over all traits), the (weighted) counts of
    branches starting without the trait, `num_trials`, and of gains,
    `num_successes`, and their mean `length` (zero without branches),
    calibrated against neutral traits if `calibration_neighborhoods` is
    given (see `tgs_calc_calibrated_pendant_lengths`).
    """
    num_traits, num_partitions = counts.num_leaves.shape
    lengths = np.stack(
        [
            tgs_calc_calibrated_pendant_lengths(
                counts=counts,
                trait=trait,
                neutral_traits=neutral_traits,
                calibration_neighborhoods=calibration_neighborhoods,
            )[0]
            for trait in range(num_traits)
        ],
    )
    return pd.DataFrame(
        {
            "trait": pd.Categorical(
                np.repeat(np.arange(num_traits), num_partitions),
                categories=range(num_traits),
            ),
            "num_trials": counts.num_leaves.ravel(),
            "num_successes": counts.num_carriers.ravel(),
            "length": lengths.ravel(),
        },
    )
