import typing

import numpy as np
from scipy import sparse

from ..._auxlib._calc_neighborhood_rate_ratios import (
    calc_neighborhood_rate_ratios,
)
from ._TgsCounts import TgsCounts


def tgs_calc_reference_rate_ratios(
    *,
    calibration_neighborhoods: sparse.spmatrix,
    counts: TgsCounts,
    refs: typing.Sequence[int],
) -> np.ndarray:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns each partition's calibration neighborhood rate ratio of gains
    among the reference traits `refs`.
    """
    return calc_neighborhood_rate_ratios(
        neighborhoods=calibration_neighborhoods,
        num_events=counts.num_carriers[refs].sum(axis=0),
        exposure=counts.pendant_length_sums[refs].sum(axis=0),
    )
