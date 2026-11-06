import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def tgs_allmdc_select_calibration_neighborhoods(
    *,
    counts: typing.Sequence[pd.DataFrame],
    blocks: typing.Sequence[typing.Tuple[int, int, np.ndarray]],
    neighborhood_fractions: typing.Sequence[float],
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    max_rate_ratio: float,
) -> sparse.csr_matrix:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_polars`. Acronym
    tgs stands for trait gain/selection.

    Returns partition-by-partition calibration neighborhoods, selected within
    each size partition's `blocks` (see
    `tgs_allmdc_partition_by_rank_within_sizes`) from its slice of
    per-partition `counts` (see `tgs_select_calibration_neighborhoods`), so
    that neighborhoods never mix size partitions, as neutral gains per unit
    length may differ with clade size.
    """
    return sparse.block_diag(
        [
            tgs_select_calibration_neighborhoods(
                partition_coords=coords,
                neighborhood_fractions=neighborhood_fractions,
                counts=[
                    trait_counts.iloc[offset : offset + num].reset_index(
                        drop=True
                    )
                    for trait_counts in counts
                ],
                neutral_traits=neutral_traits,
                trait_names=trait_names,
                max_rate_ratio=max_rate_ratio,
            )
            for offset, num, coords in blocks
        ],
        format="csr",
    )
