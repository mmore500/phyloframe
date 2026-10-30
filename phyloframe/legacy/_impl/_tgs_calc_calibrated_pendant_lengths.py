import typing

import numpy as np
from scipy import sparse

from ..._auxlib._calc_neighborhood_rate_ratios import (
    calc_neighborhood_rate_ratios,
)
from ._TgsCounts import TgsCounts
from ._tgs_find_reference_traits import tgs_find_reference_traits


def tgs_calc_calibrated_pendant_lengths(
    *,
    counts: TgsCounts,
    trait: int,
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns `trait`'s mean pendant length in each partition (zero where it
    has no leaves) and the factors calibrating it (None without
    `calibration_neighborhoods`). Calibration rescales each partition's
    pendant length by the rate ratio of gains among `trait`'s neutral
    reference within the partition's neighborhood.
    """
    num_leaves = counts.num_leaves[trait]
    pendant_lengths = np.divide(
        counts.pendant_length_sums[trait],
        num_leaves,
        out=np.zeros(num_leaves.shape),
        where=num_leaves > 0,
    )
    if calibration_neighborhoods is None:
        return pendant_lengths, None

    refs = tgs_find_reference_traits(
        neutral_traits=neutral_traits, trait=trait
    )
    factors = calc_neighborhood_rate_ratios(
        neighborhoods=calibration_neighborhoods,
        num_events=counts.num_carriers[refs].sum(axis=0),
        exposure=counts.pendant_length_sums[refs].sum(axis=0),
    )
    return pendant_lengths * factors, factors
